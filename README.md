# LangChain으로 Llama 2에서 Chatbot 만들기


## case: LangChain 

LangChain을 이용해서 Llama 2에 연결하는 경우에 아래와 같이 endpoint_kwargs에 CustomAttributes를 추가합니다. 

```python
endpoint_name = os.environ.get('endpoint')

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "inputs" : 
            [
                [
                    {
                        "role" : "system",
                        "content" : "You are a kind robot."
                    },
                    {
                        "role" : "user", 
                        "content" : prompt
                    }
                ]
            ],
            "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "max_new_tokens": 256, 
    "top_p": 0.9, 
    "temperature": 0.6
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)
```

## case: SageMaker Endpoint

SageMaker Endpoint를 직접 호출하여 prompt 응답을 받는 함수입니다.

```python
def get_llm(text):
    dialog = [{"role": "user", "content": text}]

    parameters = {
        "max_new_tokens": 256, 
        "top_p": 0.9, 
        "temperature": 0.6
    } 

    payload = {
        "inputs": [dialog], 
        "parameters":parameters
    }
    
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType='application/json', 
        Body=json.dumps(payload).encode('utf-8'),
        CustomAttributes="accept_eula=true",
    )                

    body = response["Body"].read().decode("utf8")
    body_resp = json.loads(body)
    print(body_resp[0]['generation']['content'])

    return body_resp[0]['generation']['content']
```

## Reference 

[Fundamentals of combining LangChain and Amazon SageMaker (with Llama 2 Example)](https://medium.com/@ryanlempka/fundamentals-of-combining-langchain-and-sagemaker-with-a-llama-2-example-694924ab0d92)
