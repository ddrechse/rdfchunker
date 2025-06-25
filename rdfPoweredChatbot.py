"""
RDF-Powered Vector Chatbot with Duplicate Detection

PURPOSE:
This is a complete end-to-end system that transforms RDF knowledge graphs into a
vector-powered chatbot with built-in duplicate detection and debugging capabilities.
It extracts chunks from RDF graphs, creates vector embeddings, stores them in Oracle
database, and provides an intelligent chatbot interface with comprehensive duplicate
monitoring throughout the entire pipeline.

WHAT IT DOES:
1. RDF Processing - Extracts chunks and metadata from knowledge graphs via SPARQL
2. Vector Generation - Creates embeddings using HuggingFace sentence transformers
3. Database Storage - Stores vectors with rich metadata in Oracle 23ai vector tables
4. Duplicate Detection - Real-time identification and reporting of duplicate content
5. Chatbot Creation - Builds LLM-powered assistant with semantic relationship awareness
6. Debug Monitoring - Comprehensive logging of retrieval process and content quality

KEY FEATURES:
- Semantic relationship preservation from RDF graphs
- Robust LOB handling for Oracle database compatibility
- Real-time duplicate detection during vector retrieval
- Detailed debug output showing exactly what content is sent to LLM
- Fallback retrieval methods for maximum reliability
- Rich metadata integration for context-aware responses

HOW TO USE:
1. Ensure you have a clean RDF graph file (.nt format)
2. Configure database connection parameters for your Oracle instance
3. Run process_rdf_documents() to build the complete system
4. Use the returned chain for interactive question-answering
5. Monitor debug output to verify no duplicates in retrieval results

OUTPUT:
- Fully functional vector-powered chatbot
- Comprehensive duplicate detection reports during retrieval
- Detailed logs showing content processing and LLM interaction
- Performance statistics and system health indicators

TYPICAL WORKFLOW:
generate clean RDF → process_rdf_documents() → interactive chatbot + monitoring

This system is production-ready and includes all necessary safeguards for handling
duplicate content, LOB issues, and providing transparent debugging information.
Perfect for deploying RDF knowledge graphs as intelligent chatbot applications.
"""
import array
import oracledb
import os
import json
import numpy as np
from rdflib import Graph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_nomic import NomicEmbeddings
from langchain.schema import Document


# RDF Graph Configuration
RDF_FILE = "vectorsearchcleanedchunked.nt"
CHUNK_SIZE = 1000  # This matches your RDF chunking
CHUNK_OVERLAP = 20  # Not used since RDF chunks are pre-made

VECTOR_TABLE = "rdf_vector_chunks"

### EMBEDDING MODELS
### HUGGINGFACE
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
### NOMIC
#EMBEDDING_MODEL = "nomic-embed-text-v1.5"
#VECTOR_DIMENSION = 768

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def extract_chunks_from_rdf(rdf_file):
    """Extract chunks and their metadata from RDF graph"""
    print(f"🔍 Loading RDF graph from {rdf_file}")
    
    # Load the RDF graph
    g = Graph()
    g.parse(rdf_file, format="nt")
    
    print(f"✓ Loaded RDF graph with {len(g)} triples")
    
    # SPARQL query to get all chunks with their metadata
    chunks_query = """
    PREFIX ex: <https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT ?chunk_uri ?chunk_text ?chunk_index ?chunk_size ?section_title ?section_content ?word_count
    WHERE {
        ?section_uri dc:title ?section_title .
        ?section_uri ex:hasWordCount ?word_count .
        ?section_uri ex:hasContent ?section_content .
        ?section_uri ex:hasChunk ?chunk_uri .
        ?chunk_uri ex:hasText ?chunk_text .
        ?chunk_uri ex:hasChunkIndex ?chunk_index .
        ?chunk_uri ex:hasChunkSize ?chunk_size .
    }
    ORDER BY ?section_title ?chunk_index
    """
    
    # Get relationships for each section
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
    
    print("📊 Extracting chunks and metadata...")
    
    # Execute queries
    chunk_results = list(g.query(chunks_query))
    relationship_results = list(g.query(relationships_query))
    
    # Build relationships map
    relationships_map = {}
    for row in relationship_results:
        section1 = str(row[0])
        relationship = str(row[1]).split('#')[-1]
        section2 = str(row[2])
        
        if section1 not in relationships_map:
            relationships_map[section1] = {}
        if relationship not in relationships_map[section1]:
            relationships_map[section1][relationship] = []
        relationships_map[section1][relationship].append(section2)
    
    # Process chunks
    chunks_data = []
    for row in chunk_results:
        chunk_uri = str(row[0])
        chunk_text = str(row[1])
        chunk_index = int(row[2])
        chunk_size = int(row[3])
        section_title = str(row[4])
        section_content = str(row[5])
        word_count = int(row[6])
        
        # Get relationships for this section
        section_relationships = relationships_map.get(section_title, {})
        
        # Create rich metadata
        metadata = {
            "chunk_uri": chunk_uri,
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
            "section_title": section_title,
            "section_word_count": word_count,
            "section_summary": section_content[:200] + "..." if len(section_content) > 200 else section_content,
            "relationships": section_relationships,
            "source": "Oracle AI Vector Search User Guide",
            "document_type": "technical_documentation"
        }
        
        # Create Document object (compatible with LangChain)
        document = Document(
            page_content=chunk_text,
            metadata=metadata
        )
        
        chunks_data.append(document)
    
    print(f"✅ Extracted {len(chunks_data)} chunks from RDF graph")
    print(f"📈 Relationships found: {sum(len(rels) for rels in relationships_map.values())}")
    
    # Show sample chunk
    if chunks_data:
        sample = chunks_data[0]
        print(f"\n📋 Sample chunk:")
        print(f"  Section: {sample.metadata['section_title']}")
        print(f"  Content: {sample.page_content[:100]}...")
        print(f"  Relationships: {len(sample.metadata['relationships'])} types")
    
    return chunks_data

def create_enhanced_vector_table(connection, table_name):
    """Create enhanced table with RDF metadata support (OracleVS compatible)"""
    print(f"🏗️  Creating enhanced vector table: {table_name}")
    
    with connection.cursor() as cursor:
        # First, drop table if it exists to ensure clean state
        try:
            cursor.execute(f"DROP TABLE {table_name}")
            print(f"✓ Dropped existing table {table_name}")
        except:
            print(f"📝 Table {table_name} didn't exist (this is normal for first run)")
        
        # Create the table with exact OracleVS expected schema
        cursor.execute(f"""
            CREATE TABLE {table_name} (
                id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY,
                embedding VECTOR({VECTOR_DIMENSION}),
                text CLOB,
                metadata CLOB,
                chunk_uri VARCHAR2(500),
                chunk_index NUMBER,
                chunk_size NUMBER,
                section_title VARCHAR2(500),
                section_word_count NUMBER,
                section_summary CLOB,
                relationships CLOB,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT {table_name}_pk PRIMARY KEY (id)
            )
        """)
        print(f"✅ Created table {table_name}")
        
        # Verify the table structure
        cursor.execute(f"""
            SELECT column_name, data_type 
            FROM user_tab_columns 
            WHERE table_name = UPPER('{table_name}')
            ORDER BY column_id
        """)
        
        columns = cursor.fetchall()
        print(f"📋 Table structure verification:")
        for col_name, col_type in columns:
            print(f"  - {col_name}: {col_type}")
        
        # Verify TEXT column exists
        text_column_exists = any(col[0] == 'TEXT' for col in columns)
        if not text_column_exists:
            raise Exception("ERROR: TEXT column was not created properly!")
        else:
            print(f"✅ TEXT column verified")
        
        # Create indexes for better search performance
        try:
            cursor.execute(f"CREATE INDEX {table_name}_section_idx ON {table_name} (section_title)")
            cursor.execute(f"CREATE INDEX {table_name}_chunk_idx ON {table_name} (chunk_index)")
            print("✓ Created performance indexes")
        except Exception as e:
            print(f"📝 Index creation note: {e}")
        
        connection.commit()

def insert_rdf_chunks_with_vectors(connection, chunks_data, embeddings_model, table_name):
    """Insert RDF chunks with embeddings and rich metadata"""
    print(f"🚀 Processing {len(chunks_data)} chunks for vector insertion...")
    
    # First, verify the table structure before inserting
    with connection.cursor() as cursor:
        cursor.execute(f"""
            SELECT column_name 
            FROM user_tab_columns 
            WHERE table_name = UPPER('{table_name}')
            AND column_name IN ('TEXT', 'METADATA', 'EMBEDDING')
        """)
        required_columns = [row[0] for row in cursor.fetchall()]
        print(f"📋 Required columns found: {required_columns}")
        
        if 'TEXT' not in required_columns:
            raise Exception(f"CRITICAL: TEXT column missing from {table_name}!")
        if 'METADATA' not in required_columns:
            raise Exception(f"CRITICAL: METADATA column missing from {table_name}!")
        if 'EMBEDDING' not in required_columns:
            raise Exception(f"CRITICAL: EMBEDDING column missing from {table_name}!")
    
    # Extract texts for embedding
    texts = [chunk.page_content for chunk in chunks_data]
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embeddings = embeddings_model.embed_documents(texts)
    embeddings = [np.asarray(emb, dtype=np.float32) for emb in embeddings]
    
    # Optionally apply L2 normalization
    # embeddings = [l2_normalize(emb) for emb in embeddings]
    
    print(f"✓ Generated embeddings: {len(embeddings)} x {len(embeddings[0])}")
    
    # Prepare data for insertion
    oracle_vectors = [array.array('f', emb) for emb in embeddings]
    
    insert_data = []
    for i, (chunk, vector) in enumerate(zip(chunks_data, oracle_vectors)):
        metadata = chunk.metadata
        
        # Create enhanced metadata JSON with all RDF information
        enhanced_metadata = {
            "chunk_uri": metadata['chunk_uri'],
            "chunk_index": metadata['chunk_index'],
            "chunk_size": metadata['chunk_size'],
            "section_title": metadata['section_title'],
            "section_word_count": metadata['section_word_count'],
            "section_summary": metadata['section_summary'],
            "relationships": metadata['relationships'],
            "source": metadata.get('source', 'Oracle AI Vector Search User Guide'),
            "document_type": metadata.get('document_type', 'technical_documentation')
        }
        
        insert_data.append((
            vector,                                    # embedding
            chunk.page_content,                       # text (OracleVS expects this column name)
            json.dumps(enhanced_metadata),            # metadata (JSON string for OracleVS)
            metadata['chunk_uri'],                    # chunk_uri
            metadata['chunk_index'],                  # chunk_index
            metadata['chunk_size'],                   # chunk_size
            metadata['section_title'],                # section_title
            metadata['section_word_count'],           # section_word_count
            metadata['section_summary'],              # section_summary
            json.dumps(metadata['relationships'])     # relationships (JSON string)
        ))
    
    # Insert in batches (smaller batches for large datasets)
    print("💾 Inserting chunks into database...")
    batch_size = 100  # Process in smaller batches
    total_inserted = 0
    
    with connection.cursor() as cursor:
        for i in range(0, len(insert_data), batch_size):
            batch = insert_data[i:i + batch_size]
            try:
                cursor.executemany(f"""
                    INSERT INTO {table_name} (
                        embedding, text, metadata, chunk_uri, chunk_index, chunk_size,
                        section_title, section_word_count, section_summary, relationships
                    ) VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10)
                """, batch)
                connection.commit()
                total_inserted += len(batch)
                print(f"  ✓ Inserted batch {i//batch_size + 1}: {total_inserted}/{len(insert_data)} chunks")
            except Exception as e:
                print(f"  ❌ Error in batch {i//batch_size + 1}: {e}")
                # Try to show the exact SQL being executed
                print(f"  🔍 Sample data types: {[type(x) for x in batch[0]]}")
                raise
    
    print(f"✅ Successfully inserted {total_inserted} chunks with vectors")

def create_enhanced_retriever(connection, embeddings_model, table_name):
    """Create enhanced retriever with RDF metadata support and LOB handling"""
    print("🔍 Setting up enhanced vector retriever...")
    
    # Create a fresh connection for the vector store to avoid LOB issues

    # Connect to ADB
    # fresh_connection = oracledb.connect(
    #     user="admin",
    #     password="Password",
    #     dsn="mydb_high",
    #     config_dir="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_location="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_password="Password"
    # )

    # Connect to local Container
    fresh_connection = oracledb.connect(
        user="testuser",
        password="Password",
        dsn="localhost:1521/FREEPDB1"
    )
    
    # Custom vector store with metadata support
    vector_store = OracleVS(
        client=fresh_connection,
        embedding_function=embeddings_model,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    # Enhanced retriever with more results for reranking
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    
    return retriever, fresh_connection

def create_enhanced_chat_chain(retriever):
    """Create enhanced chat chain with standard retriever"""
    print("🤖 Setting up enhanced chat chain with standard retrieval...")
    
    llm = ChatOllama(model="llama3.1:latest")
    
    # Enhanced prompt that uses RDF relationships
    prompt = ChatPromptTemplate.from_template("""
You are an expert Oracle AI Vector Search assistant. Use the provided context to answer questions accurately and comprehensively.

CONTEXT FROM ORACLE VECTOR SEARCH DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If you see related sections mentioned, reference them naturally
3. Provide practical, actionable information
4. If the context mentions prerequisites or related topics, include those recommendations
5. Use technical terms accurately as they appear in Oracle documentation

QUESTION: {question}

ANSWER:""")
    
    def format_context_with_relationships(docs):
        """Format retrieved documents with relationship information and detect duplicates"""
        print(f"\n📋 FORMATTING CONTEXT FROM {len(docs)} DOCUMENTS:")
        print("=" * 60)
        
        # Duplicate detection
        seen_content = {}
        unique_docs = []
        duplicates_found = 0
        
        for i, doc in enumerate(docs):
            content = doc.page_content
            content_hash = hash(content[:200])  # Hash first 200 chars for comparison
            
            if content_hash in seen_content:
                duplicates_found += 1
                original_idx = seen_content[content_hash]
                print(f"\n🔄 DUPLICATE DETECTED:")
                print(f"   Document {i+1} is duplicate of Document {original_idx+1}")
                print(f"   Content preview: {content[:100]}...")
                continue
            else:
                seen_content[content_hash] = i
                unique_docs.append(doc)
        
        if duplicates_found > 0:
            print(f"\n⚠️  FOUND {duplicates_found} DUPLICATES - Using {len(unique_docs)} unique documents")
        else:
            print(f"\n✅ NO DUPLICATES FOUND - All {len(docs)} documents are unique")
        
        formatted_context = []
        
        for i, doc in enumerate(unique_docs, 1):
            content = doc.page_content
            
            # Extract metadata from the document
            try:
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                else:
                    metadata = {}
            except:
                metadata = {}
            
            # Add section context
            section_title = metadata.get('section_title', 'Unknown Section')
            chunk_index = metadata.get('chunk_index', 'Unknown')
            chunk_uri = metadata.get('chunk_uri', 'Unknown')
            distance = metadata.get('distance', 'Unknown')
            
            print(f"\n📄 UNIQUE DOCUMENT {i}:")
            print(f"   Section: {section_title}")
            print(f"   Chunk Index: {chunk_index}")
            print(f"   Chunk URI: {chunk_uri}")
            print(f"   Similarity Distance: {distance}")
            print(f"   Content Length: {len(content)} characters")
            print(f"   Content Preview: {content[:100]}...")
            
            context_part = f"[Section: {section_title}]\n{content}"
            
            # Add relationship hints if available
            relationships = metadata.get('relationships', {})
            if relationships and isinstance(relationships, dict):
                rel_hints = []
                for rel_type, related_sections in relationships.items():
                    if related_sections and isinstance(related_sections, list):
                        rel_hints.append(f"{rel_type}: {', '.join(related_sections[:2])}")
                
                if rel_hints:
                    context_part += f"\n[Related: {'; '.join(rel_hints)}]"
                    print(f"   Relationships: {'; '.join(rel_hints)}")
            
            formatted_context.append(context_part)
        
        final_context = "\n\n".join(formatted_context)
        
        print(f"\n📝 FINAL CONTEXT STATS:")
        print(f"   Original documents: {len(docs)}")
        print(f"   Unique documents: {len(unique_docs)}")
        print(f"   Duplicates removed: {duplicates_found}")
        print(f"   Final context length: {len(final_context)} characters")
        print("=" * 60)
        
        return final_context
    
    # Build chain with standard retrieval
    class StandardChain:
        def __init__(self, retriever, prompt, llm, formatter):
            self.retriever = retriever
            self.prompt = prompt
            self.llm = llm
            self.formatter = formatter
        
        def invoke(self, input_dict):
            question = input_dict["question"]
            print(f"\n🔍 PROCESSING QUESTION: {question}")
            print("-" * 60)
            
            docs = self.retriever.invoke(question)
            
            # Check if we actually retrieved any documents
            if not docs:
                print("❌ NO DOCUMENTS RETRIEVED - CANNOT ANSWER")
                return "❌ I couldn't find any relevant information in the Oracle Vector Search documentation to answer your question. This might be due to technical issues with data retrieval. Please try rephrasing your question or check the system logs."
            
            print(f"✅ RETRIEVED {len(docs)} DOCUMENTS FOR CONTEXT")
            
            context = self.formatter(docs)
            
            # Add a note about how many documents were used
            context_header = f"[Found {len(docs)} relevant sections from Oracle Vector Search documentation]\n\n"
            full_context = context_header + context
            
            print(f"\n📤 SENDING TO LLM:")
            print(f"   Question: {question}")
            print(f"   Context Length: {len(full_context)} characters")
            print(f"   Context Preview (first 300 chars):")
            print(f"   {full_context[:300]}...")
            print("-" * 60)
            
            prompt_value = self.prompt.format(context=full_context, question=question)
            
            print(f"\n🤖 FULL PROMPT BEING SENT TO LLM:")
            print("=" * 80)
            print(prompt_value)
            print("=" * 80)
            
            response = self.llm.invoke(prompt_value)
            
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            print(f"\n✅ LLM RESPONSE RECEIVED:")
            print(f"   Response Length: {len(final_answer)} characters")
            print(f"   Response Preview: {final_answer[:200]}...")
            
            return final_answer
    
    return StandardChain(retriever, prompt, llm, format_context_with_relationships)

def create_robust_retriever(connection, embeddings_model, table_name):
    """Create a more robust retriever that handles LOB issues"""
    print("🔍 Setting up robust vector retriever...")
    
    def robust_similarity_search(query, k=10):
        """Custom similarity search that avoids LOB issues entirely"""
        try:
            # Generate query embedding
            query_embedding = embeddings_model.embed_query(query)
            query_vector = array.array('f', np.asarray(query_embedding, dtype=np.float32))
            
            # Two-step approach to avoid LOB issues:
            # Step 1: Get IDs and distances only
            with connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        id,
                        section_title,
                        VECTOR_DISTANCE(embedding, :embedding, COSINE) as distance
                    FROM {table_name}
                    ORDER BY VECTOR_DISTANCE(embedding, :embedding, COSINE)
                    FETCH FIRST :k ROWS ONLY
                """, {
                    'embedding': query_vector,
                    'k': k
                })
                
                id_distance_pairs = cursor.fetchall()
                print(f"✓ Found {len(id_distance_pairs)} similar chunks")
            
            # Step 2: Get text content for each ID separately
            results = []
            for doc_id, section_title, distance in id_distance_pairs:
                try:
                    with connection.cursor() as text_cursor:
                        # Read CLOB content properly to avoid LOB object issues
                        text_cursor.execute(f"""
                            SELECT 
                                DBMS_LOB.SUBSTR(text, 4000, 1) as text_content,
                                chunk_uri,
                                chunk_index,
                                chunk_size,
                                section_word_count
                            FROM {table_name}
                            WHERE id = :doc_id
                        """, {'doc_id': doc_id})
                        
                        text_row = text_cursor.fetchone()
                        if text_row:
                            text_content, chunk_uri, chunk_index, chunk_size, section_word_count = text_row
                            
                            # Create simplified metadata (avoid JSON parsing of LOB)
                            metadata = {
                                'section_title': section_title,
                                'chunk_uri': chunk_uri or f"chunk_{doc_id}",
                                'chunk_index': chunk_index or 0,
                                'chunk_size': chunk_size or len(text_content),
                                'section_word_count': section_word_count or 0,
                                'distance': float(distance),
                                'relationships': {}  # Empty for now to avoid LOB issues
                            }
                            
                            # Create Document object
                            doc = Document(
                                page_content=text_content,
                                metadata=metadata
                            )
                            results.append(doc)
                            
                except Exception as e:
                    print(f"⚠️  Skipped chunk {doc_id}: {e}")
                    continue
            
            print(f"✓ Successfully retrieved {len(results)} documents for query: '{query[:50]}...'")
            return results
                
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            # Return empty list instead of failing completely
            return []
    
    return robust_similarity_search

def create_enhanced_chat_chain_robust(retriever_func):
    """Create enhanced chat chain with robust retriever function"""
    print("🤖 Setting up enhanced chat chain with robust retrieval...")
    
    llm = ChatOllama(model="llama3.1:latest")
    
    # Enhanced prompt that uses RDF relationships
    prompt = ChatPromptTemplate.from_template("""
You are an expert Oracle AI Vector Search assistant. Use the provided context to answer questions accurately and comprehensively.

CONTEXT FROM ORACLE VECTOR SEARCH DOCUMENTATION:
{context}

INSTRUCTIONS:
1. Answer based primarily on the provided context
2. If you see related sections mentioned, reference them naturally
3. Provide practical, actionable information
4. If the context mentions prerequisites or related topics, include those recommendations
5. Use technical terms accurately as they appear in Oracle documentation

QUESTION: {question}

ANSWER:""")
    
    def format_context_with_relationships(docs):
        """Format retrieved documents with relationship information and detect duplicates"""
        print(f"\n📋 FORMATTING CONTEXT FROM {len(docs)} DOCUMENTS:")
        print("=" * 60)
        
        # Duplicate detection
        seen_content = {}
        unique_docs = []
        duplicates_found = 0
        
        for i, doc in enumerate(docs):
            content = doc.page_content
            content_hash = hash(content[:200])  # Hash first 200 chars for comparison
            
            if content_hash in seen_content:
                duplicates_found += 1
                original_idx = seen_content[content_hash]
                print(f"\n🔄 DUPLICATE DETECTED:")
                print(f"   Document {i+1} is duplicate of Document {original_idx+1}")
                print(f"   Content preview: {content[:100]}...")
                continue
            else:
                seen_content[content_hash] = i
                unique_docs.append(doc)
        
        if duplicates_found > 0:
            print(f"\n⚠️  FOUND {duplicates_found} DUPLICATES - Using {len(unique_docs)} unique documents")
        else:
            print(f"\n✅ NO DUPLICATES FOUND - All {len(docs)} documents are unique")
        
        formatted_context = []
        
        for i, doc in enumerate(unique_docs, 1):
            content = doc.page_content
            
            # Extract metadata from the document
            try:
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata = doc.metadata
                else:
                    metadata = {}
            except:
                metadata = {}
            
            # Add section context
            section_title = metadata.get('section_title', 'Unknown Section')
            chunk_index = metadata.get('chunk_index', 'Unknown')
            chunk_uri = metadata.get('chunk_uri', 'Unknown')
            distance = metadata.get('distance', 'Unknown')
            
            print(f"\n📄 UNIQUE DOCUMENT {i}:")
            print(f"   Section: {section_title}")
            print(f"   Chunk Index: {chunk_index}")
            print(f"   Chunk URI: {chunk_uri}")
            print(f"   Similarity Distance: {distance}")
            print(f"   Content Length: {len(content)} characters")
            print(f"   Content Preview: {content[:100]}...")
            
            context_part = f"[Section: {section_title}]\n{content}"
            
            # Add relationship hints if available
            relationships = metadata.get('relationships', {})
            if relationships and isinstance(relationships, dict):
                rel_hints = []
                for rel_type, related_sections in relationships.items():
                    if related_sections and isinstance(related_sections, list):
                        rel_hints.append(f"{rel_type}: {', '.join(related_sections[:2])}")
                
                if rel_hints:
                    context_part += f"\n[Related: {'; '.join(rel_hints)}]"
                    print(f"   Relationships: {'; '.join(rel_hints)}")
            
            formatted_context.append(context_part)
        
        final_context = "\n\n".join(formatted_context)
        
        print(f"\n📝 FINAL CONTEXT STATS:")
        print(f"   Original documents: {len(docs)}")
        print(f"   Unique documents: {len(unique_docs)}")
        print(f"   Duplicates removed: {duplicates_found}")
        print(f"   Final context length: {len(final_context)} characters")
        print("=" * 60)
        
        return final_context
    
    # Build chain with robust retrieval
    class RobustChain:
        def __init__(self, retriever_func, prompt, llm, formatter):
            self.retriever_func = retriever_func
            self.prompt = prompt
            self.llm = llm
            self.formatter = formatter
        
        def invoke(self, input_dict):
            question = input_dict["question"]
            print(f"\n🔍 PROCESSING QUESTION: {question}")
            print("-" * 60)
            
            docs = self.retriever_func(question, k=10)
            
            # Check if we actually retrieved any documents
            if not docs:
                print("❌ NO DOCUMENTS RETRIEVED - CANNOT ANSWER")
                return "❌ I couldn't find any relevant information in the Oracle Vector Search documentation to answer your question. This might be due to technical issues with data retrieval. Please try rephrasing your question or check the system logs."
            
            print(f"✅ RETRIEVED {len(docs)} DOCUMENTS FOR CONTEXT")
            
            context = self.formatter(docs)
            
            # Add a note about how many documents were used
            context_header = f"[Found {len(docs)} relevant sections from Oracle Vector Search documentation]\n\n"
            full_context = context_header + context
            
            print(f"\n📤 SENDING TO LLM:")
            print(f"   Question: {question}")
            print(f"   Context Length: {len(full_context)} characters")
            print(f"   Context Preview (first 300 chars):")
            print(f"   {full_context[:300]}...")
            print("-" * 60)
            
            prompt_value = self.prompt.format(context=full_context, question=question)
            
            print(f"\n🤖 FULL PROMPT BEING SENT TO LLM:")
            print("=" * 80)
            print(prompt_value)
            print("=" * 80)
            
            response = self.llm.invoke(prompt_value)
            
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            print(f"\n✅ LLM RESPONSE RECEIVED:")
            print(f"   Response Length: {len(final_answer)} characters")
            print(f"   Response Preview: {final_answer[:200]}...")
            
            return final_answer
    
    return RobustChain(retriever_func, prompt, llm, format_context_with_relationships)

# Add this alternative method to the main function
def process_rdf_documents():
    """Main function to process RDF graph and create vector chatbot"""
    print("🚀 STARTING RDF-POWERED VECTOR CHATBOT SETUP")
    print("=" * 60)
    
    # Connect to Oracle
    print("🔌 Connecting to Oracle Database...")
    # Connect to ADB
    # connection = oracledb.connect(
    #     user="admin",
    #     password="Password",
    #     dsn="mydb_high",
    #     config_dir="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_location="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_password="Password"
    # )

    # Connect to local Container
    connection = oracledb.connect(
        user="testuser",
        password="Password",
        dsn="localhost:1521/FREEPDB1"
    )
    print("✅ Connected to Oracle Database")
    
    try:
        # Extract chunks from RDF
        chunks_data = extract_chunks_from_rdf(RDF_FILE)
        
        # Set up embedding model
        print("🧠 Initializing embedding model...")
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print(f"✅ Using {EMBEDDING_MODEL}")
        
        # Create enhanced table
        create_enhanced_vector_table(connection, VECTOR_TABLE)
        
        # Insert chunks with vectors and metadata
        insert_rdf_chunks_with_vectors(connection, chunks_data, embeddings_model, VECTOR_TABLE)
        
        # Try both retrieval methods
        print("\n🔄 Setting up retrieval methods...")
        
        # Method 1: Try OracleVS (might have LOB issues)
        try:
            retriever, retriever_connection = create_enhanced_retriever(connection, embeddings_model, VECTOR_TABLE)
            chain = create_enhanced_chat_chain(retriever)
            print("✅ OracleVS retriever created successfully")
            use_robust = False
        except Exception as e:
            print(f"⚠️  OracleVS retriever failed: {e}")
            use_robust = True
        
        # Method 2: Robust custom retriever (handles LOB issues)
        if use_robust:
            print("🔄 Using robust custom retriever...")
            robust_retriever = create_robust_retriever(connection, embeddings_model, VECTOR_TABLE)
            chain = create_enhanced_chat_chain_robust(robust_retriever)
            retriever_connection = connection
        
        # Test the system
        print("\n" + "=" * 60)
        print("🧪 TESTING RDF-POWERED CHATBOT")
        print("=" * 60)
        
        # test_questions = [
        #     "What are Vector Indexes?",
        #     "How do I create vector embeddings?",
        #     "What is the relationship between vectors and similarity search?",
        #     "What are the performance considerations for vector indexes?"
        # ]
        
        test_questions = [
            "What are Vector Indexes?",
            "What are the different index types I can use?",
            "What is a Vector embedding?"
        ]

        for i, question in enumerate(test_questions):
            print(f"\n❓ QUESTION {i+1}: {question}")
            print("-" * 50)
            try:
                answer = chain.invoke({"question": question})
                print(f"🤖 ANSWER: {answer}")
            except Exception as e:
                print(f"❌ Error answering question: {e}")
                print("🔄 This might be a temporary issue, trying to continue...")
            print("-" * 50)
        
        print(f"\n🎉 SUCCESS! RDF-powered vector chatbot is ready!")
        print(f"📊 Database table: {VECTOR_TABLE}")
        print(f"📈 Total chunks processed: {len(chunks_data)}")
        print(f"🔧 Using robust retriever: {use_robust}")
        
        # Return both connections
        return chain, connection, retriever_connection
        
    except Exception as e:
        print(f"❌ Error: {e}")
        connection.close()
        raise

if __name__ == "__main__":
    chain, connection, retriever_connection = process_rdf_documents()
    
    print(f"\n💡 Interactive Mode:")
    print(f"Use: answer = chain.invoke({{'question': 'your question here'}})")
    print(f"Main connection available as 'connection' variable")
    print(f"Retriever connection available as 'retriever_connection' variable")
    print(f"\n⚠️  Note: If you get LOB errors, the system will try to continue with other questions.")