
import os
import json
import logging

import boto3
from botocore.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


load_dotenv()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)


def get_bedrock_runtime(region_name, *args, **kwargs):
    return boto3.client(service_name='bedrock-runtime', region_name=region_name, *args, **kwargs)


def get_langchain_bedrock_llm(model_id, client, *args, **kwargs):
    return BedrockLLM(model_id=model_id, client=client, *args, **kwargs)


def convert_langchaindoc_to_dict(doc):
    return {
      'page_content': doc.page_content,
      'metadata': doc.metadata
    }


def lambda_handler(events, context):
    # get query
    logging.info(events)
    print(events)
    rawdict = False
    if isinstance(events['body'], dict):
        logging.info("dictionary")
        print("dictionary")
        query = events['body']
        rawdict = True
    else:
        logging.info("string")
        print("string")
        query = json.loads(events['body'])

    # get query question
    question = query['question']

    # retrieve config
    llm_name = query.get('llm_name', os.getenv('DEFAULTLLM'))

    # getting an instance of LLM
    llm_config = query.get('llm_config', {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.8
    })
    bedrock_runtime = get_bedrock_runtime('us-east-1', config=Config(read_timeout=1024))
    llm = get_langchain_bedrock_llm(llm_name, bedrock_runtime, config=llm_config)

    # loading the embedding model
    # the embedding model must be saved to EFS first
    embed_model_name = os.getenv('EMBEDMODELFILENAME')
    s3 = boto3.resource('s3')
    s3.Bucket(os.getenv('EMBEDMODELS3BUCKET')).download_file(embed_model_name, os.path.join('/', 'tmp', embed_model_name))
    embedding_model = GPT4AllEmbeddings(
        model_name=embed_model_name,
        gpt4all_kwargs={
            'model_path': os.path.join('/', 'tmp'),
            'allow_download': False
        }
    )

    # loading vector database
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name='mlbooks',
        prefer_grpc=True
    )
    retriever = qdrant.as_retriever()
    print(' -> Vector database loaded and retrieved initialized.')

    # getting the chain
    print('Making langchain')
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)

    # get the results
    print('Grabbing results...')
    result_json = qa.invoke({'query': question})
    print(result_json)
    result_dict = {
        'query': result_json['query'],
        'answer': result_json['result'],
        'source_documents': [convert_langchaindoc_to_dict(doc) for doc in result_json['source_documents']]
    }

    # return
    return {'statusCode': 200, 'body': result_dict if rawdict else json.dumps(result_dict)}
