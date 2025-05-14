'''
This script will perform the following tasks:

Import necessary libraries and the API key from config.py.
Initialize the embeddings model.
Define example documents.
Convert documents to embeddings and add them to a FAISS index.
Implement a simple retriever class.
Implement the RAG class combining retrieval and generation.
Run an example query.

'''

import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
import config  # Importing the config file

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Example documents 
# Provide any information in the form of text on which you want to train
documents = [ 
    '''
    Introduction: The smart building and housing market in China and the Middle East is rapidly expanding. China’s smart building market alone is projected to grow from about US$19.1 billion (2024) to $92.4 billion by 2030 (30.1% CAGR). The Middle East (with Africa) is forecast to reach $47.5 billion by 2030 (30.3% CAGR). This growth is driven by urbanization, energy-efficiency mandates, and massive smart city initiatives (e.g. UAE Vision 2030, Saudi NEOM, Masdar City). This report details the development process, market structure, offerings, and strategies for a company building smart-building IoT solutions using Chinese hardware in these markets.
Resources Required by Development Phase
Developing an IoT smart building solution involves multiple phases, each with distinct resource needs and costs:
•	R&D (Concept & Design): Requires hardware and firmware engineers, IoT/cloud software developers, and access to prototyping tools. Early R&D budgets often run into tens or hundreds of thousands USD, especially for custom sensors and connectivity (one source notes a typical minimum IoT solution development cost ~$50,000). This phase includes feasibility studies, design of system architecture, and initial software/platform development.
•	Prototyping: Involves building proof-of-concept hardware (e.g. sensor boards, gateways) and software demos. Typical IoT prototype costs start around $6,000, covering components, 3D-printed enclosures, test equipment, and third-party modules (Zigbee, NB-IoT modems, etc.). Iterative prototyping and pilot installs may push costs higher (often $10k–$30k total) depending on scale. Prototyping verifies device designs, firmware, connectivity (Wi Fi, NB IoT, etc.), and cloud integration before large-scale build.
•	Manufacturing: Scaling requires tooling, PCB fabrication, enclosure molds, and component sourcing (often from Chinese suppliers). Initial tooling and line setup can cost tens of thousands. Unit costs depend on volume: for example, a Zigbee or NB-IoT sensor module (SoC + radio + power) might cost $15–$30 each in volume, while a smart thermostat or gateway could be $30–$70. Partnering with contract manufacturers (OEM/ODM) in China can minimize costs. A small production run (hundreds of units) might require $20k–$50k in parts and assembly fees; large runs benefit from economies of scale (unit cost drops significantly beyond several thousand units).
•	Integration: Building installations need system integrators and engineers to adapt the solution to specific building environments. This phase uses skilled labor (electricians, HVAC technicians, IT staff) to install sensors, gateways, and connect to building systems (BMS, SCADA, etc.). Integration often accounts for 20–30% of total project cost. For large commercial buildings, integrators may charge $5–$15 per square foot for a full BMS installation (covering sensors, wiring, controllers), reflecting labor and configuration.
•	Deployment & Commissioning: Includes on-site setup, network provisioning (e.g. LoRaWAN gateways or NB-IoT SIMs), cloud account setup, and end-user training. These costs vary by project size but can add another 10–20% on top of hardware. For example, deploying an IoT monitoring system in a mid-sized building might incur $10k–$50k in labor and services.
•	Support and Maintenance: Ongoing services include cloud hosting, software updates, and field maintenance. Annual support is often priced as a percentage of hardware cost (e.g. 10–20% per year). Companies should budget for technical support staff or partnerships to ensure timely service.

    ''',
]

# Convert documents to embeddings
document_embeddings = embeddings.embed_documents(documents)
document_embeddings = np.array(document_embeddings).astype('float32')

# Initialize FAISS index and add document embeddings
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Define a retriever class
class SimpleRetriever:
    def __init__(self, index, documents):
        self.index = index
        self.documents = documents

    def retrieve(self, query, top_k=1):
        query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

retriever = SimpleRetriever(index, documents)

# Combine the retriever with the generator
class SimpleRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def generate(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        augmented_query = f"Context: {' '.join(retrieved_docs)} Query: {query}"
        response = self.llm.invoke(augmented_query)
        return response

llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
rag = SimpleRAG(llm, retriever)

# Example usage
query = "how to minimize costs?"
response = rag.generate(query)
print(response)