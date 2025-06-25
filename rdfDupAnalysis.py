"""
RDF Knowledge Graph Duplicate Analysis Tool

PURPOSE:
This tool performs comprehensive analysis of an existing RDF knowledge graph to identify
and quantify various types of duplicate content that may be causing issues in vector
search and chatbot applications. It examines the graph structure to pinpoint exactly
where duplicates exist and their severity.

WHAT IT ANALYZES:
1. Content Duplicates - Chunks with identical or nearly identical text content
2. URI Duplicates - Multiple chunk URIs pointing to the same content
3. Index Duplicates - Sections with multiple chunks sharing the same index numbers
4. Relationship Duplicates - Redundant semantic relationships between sections
5. Structural Issues - Problematic sections and their chunk distributions

HOW TO USE:
1. Run this AFTER generating your RDF graph to validate quality
2. Review the quantified duplicate analysis results
3. Use findings to determine if RDF regeneration is needed
4. Apply suggested fixes based on specific issues identified

OUTPUT:
- Detailed statistics on each type of duplicate found
- Sample cases showing actual duplicate patterns in your data
- Summary assessment of overall graph quality
- Specific recommendations for addressing each issue type

TYPICAL WORKFLOW:
generate RDF → analyze_rdf_duplicates() → review issues → apply fixes → regenerate

This tool helps you validate the quality of your knowledge graph and ensures
clean, non-duplicate data for optimal vector search and chatbot performance.
Use it as a quality assurance step before deploying your RDF graph to production.
"""
from rdflib import Graph
import json
from collections import defaultdict, Counter

def analyze_rdf_duplicates(rdf_file="vectorsearchuserguide.nt"):
    """Comprehensive analysis of duplicates in RDF graph"""
    print("🔍 ANALYZING RDF GRAPH FOR DUPLICATES")
    print("=" * 60)
    
    # Load the RDF graph
    g = Graph()
    g.parse(rdf_file, format="nt")
    print(f"✅ Loaded RDF graph with {len(g):,} triples")
    
    # ===== ANALYSIS 1: Check for duplicate chunks =====
    print("\n📊 ANALYSIS 1: Checking for duplicate chunks...")
    
    chunks_query = """
    PREFIX ex: <https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?chunk_uri ?chunk_text ?chunk_index ?section_title
    WHERE {
        ?section_uri dc:title ?section_title .
        ?section_uri ex:hasChunk ?chunk_uri .
        ?chunk_uri ex:hasText ?chunk_text .
        ?chunk_uri ex:hasChunkIndex ?chunk_index .
    }
    ORDER BY ?section_title ?chunk_index
    """
    
    chunk_results = list(g.query(chunks_query))
    print(f"📋 Total chunks found: {len(chunk_results)}")
    
    # Check for duplicate chunk content
    content_hash_map = {}
    content_duplicates = []
    
    for i, row in enumerate(chunk_results):
        chunk_uri = str(row[0])
        chunk_text = str(row[1])
        chunk_index = int(row[2]) if row[2] is not None else 0
        section_title = str(row[3])
        
        # Create content hash
        content_hash = hash(chunk_text[:200])  # First 200 chars
        
        if content_hash in content_hash_map:
            content_duplicates.append({
                'current': {'uri': chunk_uri, 'section': section_title, 'index': chunk_index, 'content_preview': chunk_text[:100]},
                'original': content_hash_map[content_hash]
            })
        else:
            content_hash_map[content_hash] = {
                'uri': chunk_uri, 
                'section': section_title, 
                'index': chunk_index, 
                'content_preview': chunk_text[:100]
            }
    
    print(f"🔄 Content duplicates found: {len(content_duplicates)}")
    if content_duplicates:
        print("📋 Sample content duplicates:")
        for i, dup in enumerate(content_duplicates[:5]):
            print(f"  {i+1}. Duplicate:")
            print(f"     Original: {dup['original']['section']} (index {dup['original']['index']})")
            print(f"     Duplicate: {dup['current']['section']} (index {dup['current']['index']})")
            print(f"     Content: {dup['current']['content_preview']}...")
    
    # ===== ANALYSIS 2: Check for duplicate chunk URIs =====
    print(f"\n📊 ANALYSIS 2: Checking for duplicate chunk URIs...")
    
    chunk_uri_counts = Counter(str(row[0]) for row in chunk_results)
    duplicate_uris = {uri: count for uri, count in chunk_uri_counts.items() if count > 1}
    
    print(f"🔄 Duplicate URIs found: {len(duplicate_uris)}")
    if duplicate_uris:
        print("📋 Sample duplicate URIs:")
        for i, (uri, count) in enumerate(list(duplicate_uris.items())[:5]):
            print(f"  {i+1}. {uri}: appears {count} times")
    
    # ===== ANALYSIS 3: Check section-chunk relationships =====
    print(f"\n📊 ANALYSIS 3: Analyzing section-chunk relationships...")
    
    section_chunk_map = defaultdict(list)
    for row in chunk_results:
        chunk_uri = str(row[0])
        section_title = str(row[3])
        chunk_index = int(row[2]) if row[2] is not None else 0
        
        section_chunk_map[section_title].append({
            'uri': chunk_uri,
            'index': chunk_index
        })
    
    print(f"📋 Sections with chunks: {len(section_chunk_map)}")
    
    # Check for sections with duplicate chunk indices
    sections_with_duplicate_indices = {}
    for section, chunks in section_chunk_map.items():
        indices = [chunk['index'] for chunk in chunks]
        index_counts = Counter(indices)
        duplicate_indices = {idx: count for idx, count in index_counts.items() if count > 1}
        
        if duplicate_indices:
            sections_with_duplicate_indices[section] = {
                'total_chunks': len(chunks),
                'duplicate_indices': duplicate_indices
            }
    
    print(f"🔄 Sections with duplicate indices: {len(sections_with_duplicate_indices)}")
    if sections_with_duplicate_indices:
        print("📋 Sample sections with duplicate indices:")
        for i, (section, info) in enumerate(list(sections_with_duplicate_indices.items())[:5]):
            print(f"  {i+1}. {section}: {info['total_chunks']} chunks, duplicates: {info['duplicate_indices']}")
    
    # ===== ANALYSIS 4: Check relationships =====
    print(f"\n📊 ANALYSIS 4: Analyzing relationships...")
    
    relationships_query = """
    PREFIX ex: <https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?section1_title ?relationship ?section2_title
    WHERE {
        ?section1_uri dc:title ?section1_title .
        ?section2_uri dc:title ?section2_title .
        ?section1_uri ?relationship ?section2_uri .
        FILTER(STRSTARTS(STR(?relationship), STR(ex:)))
        FILTER(?relationship != ex:hasHeader && ?relationship != ex:hasChunk && 
               ?relationship != ex:hasContent && ?relationship != ex:hasText &&
               ?relationship != ex:hasWordCount && ?relationship != ex:hasCharCount &&
               ?relationship != ex:hasSummary && ?relationship != ex:hasChunkIndex &&
               ?relationship != ex:hasChunkSize && ?relationship != ex:belongsToSection)
    }
    """
    
    rel_results = list(g.query(relationships_query))
    print(f"📋 Total relationships: {len(rel_results)}")
    
    # Analyze relationship types
    rel_type_counts = Counter()
    for row in rel_results:
        rel_type = str(row[1]).split('#')[-1]
        rel_type_counts[rel_type] += 1
    
    print(f"📋 Relationship types:")
    for rel_type, count in rel_type_counts.most_common():
        print(f"  - {rel_type}: {count}")
    
    # Check for duplicate relationships
    relationship_triples = set()
    duplicate_relationships = []
    
    for row in rel_results:
        section1 = str(row[0])
        relationship = str(row[1])
        section2 = str(row[2])
        triple = (section1, relationship, section2)
        
        if triple in relationship_triples:
            duplicate_relationships.append(triple)
        else:
            relationship_triples.add(triple)
    
    print(f"🔄 Duplicate relationship triples: {len(duplicate_relationships)}")
    
    # ===== ANALYSIS 5: Sample chunks from problematic sections =====
    print(f"\n📊 ANALYSIS 5: Examining problematic sections...")
    
    if sections_with_duplicate_indices:
        problem_section = list(sections_with_duplicate_indices.keys())[0]
        print(f"📋 Examining section: '{problem_section}'")
        
        problem_chunks_query = f"""
        PREFIX ex: <https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        
        SELECT ?chunk_uri ?chunk_text ?chunk_index ?chunk_size
        WHERE {{
            ?section_uri dc:title "{problem_section}" .
            ?section_uri ex:hasChunk ?chunk_uri .
            ?chunk_uri ex:hasText ?chunk_text .
            ?chunk_uri ex:hasChunkIndex ?chunk_index .
            ?chunk_uri ex:hasChunkSize ?chunk_size .
        }}
        ORDER BY ?chunk_index
        """
        
        problem_chunk_results = list(g.query(problem_chunks_query))
        print(f"📋 Chunks in this section: {len(problem_chunk_results)}")
        
        for i, row in enumerate(problem_chunk_results[:10]):  # Show first 10
            chunk_uri = str(row[0])
            chunk_text = str(row[1])
            chunk_index = int(row[2]) if row[2] is not None else 0
            chunk_size = int(row[3]) if row[3] is not None else 0
            
            print(f"  Chunk {i+1}: Index={chunk_index}, Size={chunk_size}")
            print(f"    URI: {chunk_uri}")
            print(f"    Content: {chunk_text[:100]}...")
    
    # ===== SUMMARY =====
    print(f"\n" + "=" * 60)
    print(f"📊 SUMMARY OF DUPLICATE ANALYSIS")
    print(f"=" * 60)
    print(f"Total RDF triples: {len(g):,}")
    print(f"Total chunks: {len(chunk_results):,}")
    print(f"Unique sections: {len(section_chunk_map)}")
    print(f"Total relationships: {len(rel_results):,}")
    print(f"")
    print(f"🚨 ISSUES FOUND:")
    print(f"  - Content duplicates: {len(content_duplicates)}")
    print(f"  - URI duplicates: {len(duplicate_uris)}")
    print(f"  - Sections with duplicate indices: {len(sections_with_duplicate_indices)}")
    print(f"  - Duplicate relationship triples: {len(duplicate_relationships)}")
    
    if any([content_duplicates, duplicate_uris, sections_with_duplicate_indices, duplicate_relationships]):
        print(f"\n⚠️  YOUR RDF GRAPH HAS DUPLICATE ISSUES!")
        print(f"This explains why your vector retrieval is finding duplicates.")
        print(f"You should regenerate your RDF graph with duplicate prevention.")
    else:
        print(f"\n✅ YOUR RDF GRAPH LOOKS CLEAN!")
        print(f"The duplicates must be coming from the retrieval process.")
    
    return {
        'content_duplicates': len(content_duplicates),
        'uri_duplicates': len(duplicate_uris),
        'sections_with_duplicate_indices': len(sections_with_duplicate_indices),
        'duplicate_relationships': len(duplicate_relationships),
        'total_chunks': len(chunk_results),
        'total_relationships': len(rel_results)
    }

def suggest_fixes():
    """Suggest fixes for common RDF duplicate issues"""
    print(f"\n🔧 SUGGESTED FIXES FOR RDF DUPLICATES:")
    print(f"=" * 60)
    print(f"")
    print(f"1. **Content Extraction Issues**:")
    print(f"   - Same PDF content extracted multiple times")
    print(f"   - Fix: Add deduplication in extract_section_content()")
    print(f"")
    print(f"2. **Chunking Issues**:")
    print(f"   - Overlapping chunks with same content")
    print(f"   - Fix: Ensure sentence boundaries don't create duplicates")
    print(f"")
    print(f"3. **Relationship Over-creation**:")
    print(f"   - Too many relationship patterns matching")
    print(f"   - Fix: Make relationship patterns more specific")
    print(f"")
    print(f"4. **Multiple RDF Generation Runs**:")
    print(f"   - Running the script multiple times without clearing")
    print(f"   - Fix: Always start with fresh RDF graph")
    print(f"")
    print(f"💡 **Quick Fix**: Regenerate your RDF graph with:")
    print(f"   - More specific header matching")
    print(f"   - Stricter duplicate detection")
    print(f"   - Fewer relationship patterns")

if __name__ == "__main__":
    results = analyze_rdf_duplicates("vectorsearchcleanedchunked.nt")
    suggest_fixes()
