# Oracle AI RDF Enabled Chatbot

A comprehensive end-to-end system for building intelligent chatbots from PDF documentation using Oracle Database 23ai's native vector search capabilities, RDF knowledge graphs, and advanced deduplication techniques.

## 🌟 Overview

This project transforms Oracle AI Vector Search documentation into a production-ready RAG (Retrieval-Augmented Generation) system by:

- **Extracting structured content** from PDF documents with intelligent deduplication
- **Building RDF knowledge graphs** with semantic relationships
- **Creating vector embeddings** optimized for Oracle Database 23ai
- **Providing AI-powered Q&A** with comprehensive duplicate detection
- **Validating system quality** through multi-dimensional analysis

## 🏗️ Architecture

```
PDF Document → RDF Graph → Vector Embeddings → Oracle Database → AI Chatbot
     ↓              ↓              ↓              ↓              ↓
Content Extraction  Knowledge     Embeddings    Vector Storage  Smart Retrieval
& Deduplication    Relationships  Generation    with Metadata   & LLM Response
```

## 📁 Project Structure

### Core Scripts

| File | Purpose | Key Features |
|------|---------|--------------|
| `createRDFGraph.py` | PDF → RDF conversion | Advanced deduplication, TOC extraction, semantic chunking |
| `oracleRAG.py` | Traditional RAG pipeline | Direct PDF processing, Oracle vector storage |
| `rdfPoweredChatbot.py` | RDF-enhanced chatbot | Knowledge graph integration, relationship-aware responses |

### Analysis & Validation Tools

| File | Purpose | Key Features |
|------|---------|--------------|
| `validateRDFGraph.py` | Quality assessment | 5-tier validation, chatbot readiness scoring |
| `rdfDupAnalysis.py` | Duplicate detection | Content analysis, URI validation, relationship audit |
| `pdfDupAnalysis.py` | Alternative analyzer | Comprehensive duplicate identification and reporting |

## 🚀 Quick Start

### Prerequisites

```bash
# Python packages
pip install PyPDF2 rdflib oracledb langchain numpy
pip install sentence-transformers langchain-community langchain-ollama

# Docker (for Oracle Database)
# Ollama with Llama model (for local LLM)
```

### Oracle Database Setup (Recommended)

The easiest way to get started is using Docker with Oracle Database 23ai Free:

```bash
# Start Oracle Database 23ai container with vector support
docker run --name free23ai -d -p 1521:1521 \
  -e ORACLE_PASSWORD=Welcome12345 \
  -e APP_USER=testuser \
  -e APP_USER_PASSWORD=Welcome12345 \
  gvenzl/oracle-free:23.7-slim-faststart
```

**That's it!** The scripts are pre-configured to work with this setup out of the box.

### Database Configuration

#### Default Configuration (Works with Docker setup above)

The scripts are pre-configured with these default settings that match the Docker container:

```python
# Connection example
connection = oracledb.connect(
    user="testuser",
    password="Welcome12345", 
    dsn="localhost:1521/FREEPDB1"
)
```

#### Alternative: Autonomous Database

If you prefer using Oracle Autonomous Database, update the connection parameters and point at the downloaded wallet:

```python
    # Connect to ADB
    connection = oracledb.connect(
         user="admin",
         password="Password",
         dsn="mydb_high",
         config_dir="/Users/dev/Wallets/Wallet_mydb",
         wallet_location="/Users/dev/Wallets/Wallet_mydb",
         wallet_password="Password"
     )
```

### Basic Usage

1. **Start Oracle Database** (if using Docker)
```bash
docker run --name free23ai -d -p 1521:1521 \
  -e ORACLE_PASSWORD=Welcome12345 \
  -e APP_USER=testuser \
  -e APP_USER_PASSWORD=Welcome12345 \
  gvenzl/oracle-free:23.7-slim-faststart

# Wait 30-60 seconds for database to fully start
```

2. **Generate RDF Knowledge Graph**
```bash
python createRDFGraph.py
# Creates: vectorsearchcleanedchunked.nt
```

3. **Validate Graph Quality**
```bash
python validateRDFGraph.py
# Outputs: Comprehensive quality assessment
```

4. **Build Vector-Powered Chatbot**
```bash
python rdfPoweredChatbot.py
# Creates: Complete RAG system with chatbot interface
```

## 🔧 Detailed Workflows

### Workflow 1: Traditional RAG (Direct PDF)

```python
# Run oracleRAG.py for direct PDF processing
python oracleRAG.py

# Features:
# - Direct PDF chunking
# - HuggingFace embeddings
# - Oracle vector storage
# - Basic similarity search
```

### Workflow 2: RDF-Enhanced RAG (Recommended)

```python
# Step 1: Create clean RDF graph
python createRDFGraph.py

# Step 2: Validate quality (optional but recommended)
python validateRDFGraph.py

# Step 3: Build enhanced chatbot
python rdfPoweredChatbot.py

# Benefits:
# - Semantic relationships preserved
# - Advanced duplicate prevention
# - Context-aware responses
# - Rich metadata integration
```

## 🧠 Key Features

### Advanced Deduplication System

- **Header-level deduplication** using normalized comparison
- **Position-based overlap detection** to prevent content reuse
- **Content similarity analysis** using sequence matching
- **Final validation** to ensure no duplicate chunks exist

### Intelligent Content Processing

- **Table of Contents extraction** for document structure
- **Conservative header variations** for robust matching
- **Natural break point detection** for optimal chunking
- **Technical content validation** with domain-specific keywords

### Oracle Database Integration

- **Native VECTOR data type** support for Oracle 23ai
- **Efficient similarity search** using cosine distance
- **LOB handling** for large text content
- **Rich metadata storage** with JSON support

### Production-Ready Features

- **Comprehensive error handling** and recovery mechanisms
- **Real-time duplicate detection** during retrieval
- **Debug monitoring** with detailed logging
- **Quality validation** with scoring system

## 📊 System Validation

The validation system provides a 5-tier assessment:

### Scoring Categories (100 points total)

1. **Basic Structure** (20 pts) - Headers, chunks, triples count
2. **Content Quality** (25 pts) - Word distribution, depth analysis  
3. **Chunk Characteristics** (20 pts) - Size optimization, consistency
4. **Semantic Relationships** (15 pts) - Connection richness, diversity
5. **Chatbot Readiness** (20 pts) - Technical coverage, actionable content

### Quality Grades

- **85%+**: 🏆 EXCELLENT - Production ready
- **70-84%**: ✅ GOOD - Minor optimizations needed
- **50-69%**: ⚠️ FAIR - Moderate improvements required
- **<50%**: ❌ POOR - Significant restructuring needed

## 🎯 Example Usage

### Interactive Chatbot Session

```python
# After running rdfPoweredChatbot.py
chain, connection, retriever_connection = process_rdf_documents()

# Ask questions
answer = chain.invoke({"question": "What are Vector Indexes?"})
print(answer)

# System provides:
# - Context from relevant document sections
# - Semantic relationship awareness
# - Duplicate detection during retrieval
# - Rich metadata for citations
```

### Sample Questions

- "What are Vector Indexes?"
- "How do I create vector embeddings?"
- "What are the different index types I can use?"
- "What are the performance considerations for vector indexes?"

## 🔍 Troubleshooting

### Common Issues

**Docker Database Not Starting**
```bash
# Check container status
docker ps -a

# View logs if container failed
docker logs free23ai

# Restart if needed
docker restart free23ai
```

**Connection Errors**
```bash
# Verify database is running
docker exec -it free23ai sqlplus testuser/Welcome12345@FREEPDB1

# Check if container ports are accessible
telnet localhost 1521
```

**LOB Object Errors**
```python
# Solution: Use robust retriever in rdfPoweredChatbot.py
# The system automatically falls back to LOB-safe methods
```

**Duplicate Content in Results**
```python
# Run analysis tools first:
python rdfDupAnalysis.py

# Then regenerate with stricter deduplication:
# Adjust similarity thresholds in createRDFGraph.py
```

**Empty RDF Graph**
```python
# Check PDF path and format
# Verify TOC extraction in createRDFGraph.py
# Review header matching patterns
```

### Performance Optimization

- **Chunk Size**: Optimal range 300-1500 characters for embeddings
- **Batch Processing**: Use batch sizes of 100 for large datasets
- **Index Creation**: Add database indexes for faster retrieval
- **Memory Management**: Monitor connection pooling for large documents

## 🔧 Configuration Options

### Embedding Models

```python
# HuggingFace (default)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384

# Nomic (alternative)
EMBEDDING_MODEL = "nomic-embed-text-v1.5" 
VECTOR_DIMENSION = 768
```

### Chunking Parameters

```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 20       # Overlap between chunks
MAX_CHUNK_SIZE = 450     # Maximum chunk size for embeddings
```

### Database Tables

```python
VECTOR_TABLE = "rdf_vector_chunks"  # RDF-enhanced system
VECTOR_TABLE = "hf_emb"            # Traditional RAG system
```

## 📚 Dependencies

### Core Requirements

- **PyPDF2**: PDF text extraction
- **rdflib**: RDF graph creation and SPARQL queries
- **oracledb**: Oracle Database connectivity
- **langchain**: Document processing and RAG pipeline
- **numpy**: Vector operations and normalization
- **sentence-transformers**: Text embedding generation

### Optional Dependencies

- **langchain-ollama**: Local LLM integration
- **langchain-nomic**: Nomic embedding support
- **difflib**: Content similarity analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Oracle Database 23ai team for native vector search capabilities
- LangChain community for RAG framework
- HuggingFace for pre-trained embedding models
- Ollama project for local LLM deployment

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the validation output for specific recommendations
3. Examine debug logs for detailed error information
4. Open an issue with relevant log excerpts and system configuration

---

**Built with ❤️ for Oracle AI Vector Search and knowledge graph enthusiasts**