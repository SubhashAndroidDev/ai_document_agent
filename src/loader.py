from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("documents/company_policy.pdf")

documents = loader.load()

print(len(documents))