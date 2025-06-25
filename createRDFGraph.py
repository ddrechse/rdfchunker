"""
RDF Graph Generator for PDF Content Extraction with Advanced Deduplication
==========================================================================

This script creates a comprehensive RDF (Resource Description Framework) graph from PDF documents,
specifically designed to extract and structure content from Oracle AI Vector Search documentation
while preventing content duplication that commonly occurs in technical documentation.

Key Features:
- Intelligent PDF parsing with Table of Contents (TOC) extraction
- Advanced deduplication system with multiple prevention layers:
  * Header-level deduplication using normalized comparison
  * Position-based overlap detection to prevent content reuse
  * Content similarity analysis using sequence matching
  * Final validation to ensure no duplicate chunks exist
- Semantic relationship mapping between document sections
- Chunked content generation optimized for chatbot/AI applications
- RDF triple generation with proper namespace bindings

Dependencies:
- PyPDF2: PDF text extraction
- rdflib: RDF graph creation and serialization
- difflib: Content similarity analysis
- re, urllib.parse: Text processing and URI encoding

Input:
- PDF file path (specifically Oracle AI Vector Search User Guide)

Output:
- RDF graph serialized as N-Triples format (.nt file)
- Structured content chunks with semantic relationships
- Comprehensive deduplication reports and validation

The script is particularly designed for technical documentation that may contain:
- Repeated section headers across multiple pages
- Overlapping content between related sections  
- Similar examples or code snippets in different contexts

"""
import PyPDF2
import re
from rdflib import Graph, URIRef, Literal, Namespace
from urllib.parse import quote
from difflib import SequenceMatcher
from collections import defaultdict

# Define namespaces
EX = Namespace("https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf#")
DC = Namespace("http://purl.org/dc/elements/1.1/")

# Load PDF
pdf_file = "/Users/DDRECHSE/projects/selectAIpy/notebooks/ai-vector-search-users-guide.pdf"
reader = PyPDF2.PdfReader(pdf_file)

# Create RDF graph
g = Graph()
g.bind("ex", EX)
g.bind("dc", DC)

# Step 1: Find the TOC page(s) - SAME AS BEFORE
toc_start = None
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "Contents" in text:
        toc_start = i
        break

if toc_start is None:
    raise ValueError("Table of Contents not found in PDF.")

print(f"Found TOC at page {toc_start + 1}")

toc_lines = []
glossary_found = False

for i in range(toc_start, toc_start + 10):
    if i >= len(reader.pages):
        break
    text = reader.pages[i].extract_text()
    if not text:
        continue
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    start = 0
    for j, line in enumerate(lines):
        if "Contents" in line:
            start = j + 1
            break
    for line in lines[start:]:
        if "Glossary" in line and not glossary_found:
            glossary_found = True
            break
        elif glossary_found:
            break
        toc_lines.append(line)
    if glossary_found:
        break

print(f"Extracted {len(toc_lines)} TOC lines (up to but not including Glossary)")
headers = []
for line in toc_lines:
    cleaned = re.sub(r'^\s*\d+\s+', '', line)
    cleaned = re.sub(r'\s*[\d-]+\s*$', '', cleaned)
    cleaned = cleaned.strip()
    if cleaned:
        headers.append(cleaned)

print(f"Headers found: {headers}")

# DUPLICATE-PREVENTION CONTENT EXTRACTION
def extract_section_content_with_deduplication(reader, headers):
    """Extract content with comprehensive duplicate prevention"""
    print("\n=== CONTENT EXTRACTION WITH DUPLICATE PREVENTION ===")
    
    # ===== FIX 1: DEDUPLICATE HEADERS FIRST =====
    print(f"📋 Original headers: {len(headers)}")
    
    # Remove exact duplicates while preserving order
    seen_headers = set()
    unique_headers = []
    duplicate_headers_removed = 0
    
    for header in headers:
        header_normalized = header.strip().lower()
        if header_normalized not in seen_headers:
            seen_headers.add(header_normalized)
            unique_headers.append(header)
        else:
            duplicate_headers_removed += 1
            print(f"  🔄 Removed duplicate header: '{header}'")
    
    print(f"✅ Deduplication: {len(unique_headers)} unique headers (-{duplicate_headers_removed} duplicates)")
    headers = unique_headers
    
    # ===== EXTRACT PDF TEXT =====
    print("📖 Extracting PDF text...")
    full_text = ""
    
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                full_text += f"\n{text.strip()}"
        except Exception as e:
            print(f"⚠️  Error extracting page {i+1}: {e}")
    
    print(f"✅ Extracted {len(full_text):,} characters from PDF")
    
    # ===== FIX 2: TRACK USED TEXT POSITIONS =====
    used_text_positions = []  # [(start, end, header), ...]
    sections = {}
    
    def positions_overlap(pos1, pos2, min_overlap=100):
        """Check if two text positions significantly overlap"""
        start1, end1 = pos1
        start2, end2 = pos2
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        overlap_length = max(0, overlap_end - overlap_start)
        return overlap_length >= min_overlap
    
    def create_conservative_header_variations(header):
        """Create fewer, more conservative header variations"""
        variations = []
        
        # Original header
        variations.append(header)
        variations.append(header.lower())
        
        # Remove "About" prefix only
        no_about = re.sub(r'^about\s+', '', header, flags=re.IGNORECASE)
        if no_about != header and len(no_about) > 5:
            variations.append(no_about)
            variations.append(no_about.lower())
        
        # Key words only (but require at least 2 words)
        key_words = re.findall(r'\b\w{4,}\b', header.lower())
        if len(key_words) >= 2:
            variations.append(' '.join(key_words[:3]))  # Max 3 key words
        
        # Remove duplicates
        unique_variations = []
        for v in variations:
            if v and len(v) >= 5 and v not in unique_variations:  # Minimum length check
                unique_variations.append(v)
        
        return unique_variations
    
    # ===== PROCESS HEADERS WITH POSITION TRACKING =====
    print(f"\nProcessing {len(headers)} headers with position deduplication...")
    
    successful_extractions = 0
    position_conflicts = 0
    content_too_similar = 0
    
    for i, header in enumerate(headers):
        print(f"\r  Processing {i+1}/{len(headers)}: {header[:40]}...", end="")
        
        variations = create_conservative_header_variations(header)
        best_content = ""
        best_position = None
        best_score = 0
        
        # Try each variation
        for variation in variations:
            if len(variation) < 5:  # Skip very short variations
                continue
                
            pattern = re.escape(variation)
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            
            for match in matches:
                start_pos = match.start()
                end_pos = min(start_pos + 1500, len(full_text))  # Smaller chunks
                
                content_after = full_text[match.end():end_pos]
                
                # Find natural break point
                lines = content_after.split('\n')
                natural_end = len(content_after)
                
                for j, line in enumerate(lines[2:], 2):
                    line = line.strip()
                    if (len(line) > 10 and 
                        (line[0].isupper() or line[0].isdigit()) and
                        not line.endswith('.') and
                        len(line.split()) < 8):  # Stricter header detection
                        natural_end = sum(len(lines[k]) + 1 for k in range(j))
                        break
                
                section_content = content_after[:natural_end].strip()
                
                if len(section_content) > 100:  # Require substantial content
                    proposed_position = (match.end(), match.end() + natural_end)
                    
                    # FIX 2: CHECK FOR POSITION CONFLICTS
                    position_conflict = False
                    for used_start, used_end, used_header in used_text_positions:
                        if positions_overlap(proposed_position, (used_start, used_end)):
                            position_conflict = True
                            print(f"\n  ⚠️  Position conflict: '{header}' vs '{used_header}'")
                            break
                    
                    if not position_conflict:
                        score = len(section_content) + (1000 - start_pos) / 1000
                        if score > best_score:
                            best_score = score
                            best_content = section_content
                            best_position = proposed_position
                        break
                    else:
                        position_conflicts += 1
        
        # FIX 3: CHECK FOR CONTENT SIMILARITY
        if best_content:
            content_too_similar_flag = False
            
            for existing_header, existing_content in sections.items():
                if len(existing_content) > 100:
                    # Check similarity of first 200 characters
                    similarity = SequenceMatcher(None, 
                                               best_content[:200].lower(), 
                                               existing_content[:200].lower()).ratio()
                    
                    if similarity > 0.85:  # Very similar content
                        print(f"\n  ⚠️  Content too similar: '{header}' vs '{existing_header}' ({similarity:.2f})")
                        content_too_similar_flag = True
                        content_too_similar += 1
                        break
            
            if not content_too_similar_flag:
                # Clean and store content
                best_content = re.sub(r'\n\s*\n', '\n\n', best_content)
                best_content = best_content.strip()
                sections[header] = best_content
                
                if best_position:
                    used_text_positions.append((best_position[0], best_position[1], header))
                
                successful_extractions += 1
            else:
                sections[header] = ""  # Store empty to track the attempt
        else:
            sections[header] = ""
    
    print()  # New line after progress
    
    # ===== SUMMARY =====
    successful = sum(1 for content in sections.values() if content)
    total_chars = sum(len(content) for content in sections.values())
    
    print(f"\n=== EXTRACTION SUMMARY WITH DEDUPLICATION ===")
    print(f"Total headers processed: {len(headers)}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {len(headers) - successful}")
    print(f"Position conflicts avoided: {position_conflicts}")
    print(f"Similar content avoided: {content_too_similar}")
    print(f"Total content extracted: {total_chars:,} characters")
    print(f"Average content per section: {total_chars // max(successful, 1):,} characters")
    
    if successful == 0:
        print("\n❌ NO CONTENT EXTRACTED!")
        print("The deduplication might be too strict. Try relaxing similarity threshold.")
    elif position_conflicts > 20 or content_too_similar > 10:
        print(f"\n⚠️  High deduplication activity detected")
        print(f"This indicates your PDF has significant content overlap")
    else:
        print(f"\n✅ CLEAN EXTRACTION COMPLETED")
        print(f"Duplicates should be eliminated from your RDF graph")
    
    return sections

# FINAL VALIDATION FUNCTION
def validate_no_duplicate_chunks(sections):
    """Final validation that no duplicate chunks exist"""
    print("\n=== FINAL DUPLICATE VALIDATION ===")
    
    all_chunks = []
    
    # Simulate chunking process
    for header, content in sections.items():
        if not content or len(content) < 30:
            continue
            
        sentences = re.split(r'[.!?]+', content)
        header_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > 1000 and current_chunk:
                header_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            header_chunks.append(current_chunk.strip())
        
        for i, chunk in enumerate(header_chunks):
            all_chunks.append({
                'header': header,
                'chunk_index': i,
                'content': chunk,
                'content_hash': hash(chunk[:200])
            })
    
    # Check for duplicate chunks
    hash_counts = defaultdict(list)
    for chunk in all_chunks:
        hash_counts[chunk['content_hash']].append(chunk)
    
    duplicates_found = sum(1 for chunks in hash_counts.values() if len(chunks) > 1)
    
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Duplicate chunk groups: {duplicates_found}")
    
    if duplicates_found > 0:
        print(f"⚠️  Sample duplicate chunks:")
        for i, (hash_val, chunks) in enumerate(list(hash_counts.items())[:3]):
            if len(chunks) > 1:
                print(f"  Duplicate group {i+1}:")
                for chunk in chunks:
                    print(f"    - {chunk['header']} (chunk {chunk['chunk_index']})")
                    print(f"      {chunk['content'][:100]}...")
    else:
        print(f"✅ NO DUPLICATE CHUNKS FOUND!")
    
    return duplicates_found == 0

# Extract content using DEDUPLICATION method
sections = extract_section_content_with_deduplication(reader, headers)

# Validate no duplicates in chunks
validate_no_duplicate_chunks(sections)

# Add headers and content to RDF
print("\n=== Adding to RDF Graph ===")
for header in headers:
    safe_header = quote(header.replace(' ', '_'), safe='')
    header_uri = EX[f"TOC_Header_{safe_header}"]
    
    g.add((EX.Document, EX.hasHeader, header_uri))
    g.add((header_uri, DC.title, Literal(header)))
    
    content = sections.get(header, "")
    if content:
        g.add((header_uri, EX.hasContent, Literal(content)))
        g.add((header_uri, EX.hasWordCount, Literal(len(content.split()))))
        g.add((header_uri, EX.hasCharCount, Literal(len(content))))
        
        summary = content[:200] + "..." if len(content) > 200 else content
        g.add((header_uri, EX.hasSummary, Literal(summary)))
        print(f"✓ Added content for: {header}")
    else:
        g.add((header_uri, EX.hasContent, Literal("")))
        g.add((header_uri, EX.hasWordCount, Literal(0)))

# Create chunks with LOWER threshold
def create_content_chunks(sections, max_chunk_size=450):
    print("\n=== Creating Content Chunks ===")
    
    chunks = {}
    total_chunks = 0
    
    for header, content in sections.items():
        if not content or len(content) < 30:  # MUCH lower threshold
            continue
            
        sentences = re.split(r'[.!?]+', content)
        header_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                header_chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            header_chunks.append(current_chunk.strip())
        
        chunks[header] = header_chunks
        total_chunks += len(header_chunks)
        
        print(f"  {header}: {len(header_chunks)} chunks")
        
        # Add chunks to RDF
        safe_header = quote(header.replace(' ', '_'), safe='')
        header_uri = EX[f"TOC_Header_{safe_header}"]
        
        for i, chunk in enumerate(header_chunks):
            chunk_uri = EX[f"Chunk_{safe_header}_{i}"]
            g.add((header_uri, EX.hasChunk, chunk_uri))
            g.add((chunk_uri, EX.hasText, Literal(chunk)))
            g.add((chunk_uri, EX.hasChunkIndex, Literal(i)))
            g.add((chunk_uri, EX.hasChunkSize, Literal(len(chunk))))
            g.add((chunk_uri, EX.belongsToSection, header_uri))
    
    print(f"Total content chunks created: {total_chunks}")
    return chunks

# Create chunks
content_chunks = create_content_chunks(sections)

# Create SAFE semantic relationships
def create_safe_semantic_relationships():
    """Create focused, high-quality relationships without explosion"""
    print("\n=== Creating SAFE Semantic Relationships ===")
    
    # MUCH more specific patterns - fewer but higher quality
    relationship_patterns = {
        'prerequisite': [
            # Only very specific, exact learning sequences
            (r'^overview of oracle ai vector search$', r'get started'),
            (r'get started.*vector', r'generate vector embeddings'),
            (r'generate.*embedding', r'create.*index'),
            (r'install.*oracle', r'configur.*database'),
            (r'^introduction to', r'.*overview'),
        ],
        'related': [
            # Only 5-6 very specific technical relationships
            (r'vector index.*memory', r'hybrid vector index.*memory'),  # Specific index types
            (r'similarity.*search', r'distance.*function'),             # Core concepts  
            (r'embedding.*model', r'vector.*dimension'),                # Core concepts
            (r'performance.*tuning.*vector', r'memory.*pool.*vector'),  # Performance topics
            (r'sql.*function.*vector', r'pl/sql.*package.*vector'),     # API groupings
        ],
        'part_of': [
            # Only clear, specific hierarchical relationships
            (r'about vector generation$', r'generate vector embeddings$'),
            (r'about sql functions.*embeddings', r'generate vector embeddings$'),
            (r'about.*neighbor.*index', r'create vector indexes'),
            (r'guidelines.*vector.*index', r'create vector indexes'),
            (r'v\$vector_index$', r'vector.*index.*views'),
            (r'example.*vector.*search', r'vector.*search.*query'),
        ],
        'implements': [
            # Only clear implementation relationships
            (r'example.*vector.*index', r'vector.*index.*concept'),
            (r'tutorial.*embedding', r'embedding.*theory'),
            (r'walkthrough.*search', r'search.*procedure'),
        ]
    }
    
    relationships_created = 0
    relationship_details = []
    max_relationships = 50  # Hard limit to prevent explosion
    
    print(f"Processing {len(headers)} headers with strict patterns...")
    
    # Much more controlled matching with early termination
    for i, header1 in enumerate(headers):
        if relationships_created >= max_relationships:
            break
            
        for j, header2 in enumerate(headers):
            if i >= j or relationships_created >= max_relationships:
                continue
                
            header1_lower = header1.lower().strip()
            header2_lower = header2.lower().strip()
            
            # Check for relationship patterns - STRICT matching
            relationship_found = False
            
            for rel_type, patterns in relationship_patterns.items():
                if relationship_found or relationships_created >= max_relationships:
                    break
                    
                for pattern1, pattern2 in patterns:
                    # Much stricter matching with word boundaries
                    match1_to_2 = (re.search(pattern1, header1_lower) and 
                                  re.search(pattern2, header2_lower))
                    match2_to_1 = (re.search(pattern2, header1_lower) and 
                                  re.search(pattern1, header2_lower))
                    
                    if match1_to_2 or match2_to_1:
                        # Additional quality check - headers must be different enough
                        if len(set(header1_lower.split()) & set(header2_lower.split())) > 2:
                            continue  # Skip if headers are too similar (likely duplicates)
                        
                        uri1 = EX[f"TOC_Header_{quote(header1.replace(' ', '_'), safe='')}"]
                        uri2 = EX[f"TOC_Header_{quote(header2.replace(' ', '_'), safe='')}"]
                        
                        if rel_type == 'prerequisite':
                            if match1_to_2:
                                g.add((uri1, EX.prerequisiteFor, uri2))
                                relationship_details.append(f"PREREQ: '{header1}' → '{header2}'")
                            else:
                                g.add((uri2, EX.prerequisiteFor, uri1))
                                relationship_details.append(f"PREREQ: '{header2}' → '{header1}'")
                        elif rel_type == 'implements':
                            if match1_to_2:
                                g.add((uri1, EX.implements, uri2))
                                relationship_details.append(f"IMPL: '{header1}' → '{header2}'")
                            else:
                                g.add((uri2, EX.implements, uri1))
                                relationship_details.append(f"IMPL: '{header2}' → '{header1}'")
                        else:
                            # Symmetric but only one direction to reduce explosion
                            g.add((uri1, EX[rel_type], uri2))
                            relationship_details.append(f"{rel_type.upper()}: '{header1}' ↔ '{header2}'")
                        
                        relationships_created += 1
                        relationship_found = True
                        print(f"  ✓ Created relationship {relationships_created}/{max_relationships}")
                        break
    
    print(f"\n✅ Safe relationships created: {relationships_created}")
    print(f"📋 Sample relationships:")
    for detail in relationship_details[:min(10, len(relationship_details))]:
        print(f"  - {detail}")
    
    if relationships_created < max_relationships:
        print(f"✅ Good! Created {relationships_created} targeted relationships (under limit of {max_relationships})")
    else:
        print(f"⚠️  Hit limit of {max_relationships} relationships")
    
    return relationships_created

# Execute the SAFE relationship creation
if headers:
    create_safe_semantic_relationships()

# Save results
g.serialize("vectorsearchcleanedchunked.nt", format="nt")
print(f"\nRDF triples written to vectorsearchcleanedchunked.nt")
print(f"Total triples: {len(g)}")

# Final summary
successful_sections = sum(1 for content in sections.values() if content)
total_chunks = sum(len(chunks) for chunks in content_chunks.values())
total_content = sum(len(content) for content in sections.values())

print(f"\n=== FINAL RESULTS ===")
print(f"Headers processed: {len(headers)}")
print(f"Sections with content: {successful_sections}")
print(f"Total content chunks: {total_chunks}")
print(f"Total content: {total_content:,} characters")
print(f"RDF triples: {len(g)}")

if total_chunks > 0:
    print("🎉 SUCCESS! You now have a CLEAN RDF graph for your chatbot!")
else:
    print("❌ Still no chunks created. The PDF might have unusual formatting.")
    print("Consider manually inspecting the PDF text structure.")
