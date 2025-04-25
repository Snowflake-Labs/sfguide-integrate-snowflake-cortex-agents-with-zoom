import requests
import json
import generate_jwt
from generate_jwt import JWTGenerator

DEBUG = False

class CortexChat:
    def __init__(self, 
            agent_url: str, 
            inference_url: str,
            search_services: list, 
            semantic_models: list,
            model: str, 
            account: str,
            user: str,
            private_key_path: str
        ):
        self.agent_url = agent_url
        self.inference_url = inference_url
        self.model = model
        self.search_service = search_services[0]
        self.support_semantic_model = semantic_models[0]
        self.supply_chain_semantic_model = semantic_models[1]
        self.account = account
        self.user = user
        self.private_key_path = private_key_path
        self.jwt = JWTGenerator(self.account, self.user, self.private_key_path).get_token()

    def data_to_answer(self, df: str) -> str:
        url = self.inference_url
        headers = {
            'X-Snowflake-Authorization-Token-Type': 'KEYPAIR_JWT',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f"Bearer {self.jwt}"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", 
                          "content": f"""Extract and explain the data in the following dataframe in natural language. 
                          Only output the explanation and nothing else. \n\n {df}"""}]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 401:  # Unauthorized - likely expired JWT
            print("JWT has expired. Generating new JWT...")
            # Generate new token
            self.jwt = JWTGenerator(self.account, self.user, self.private_key_path).get_token()
            # Retry the request with the new token
            headers["Authorization"] = f"Bearer {self.jwt}"
            print("New JWT generated. Sending new request to Cortex Inference API. Please wait...")
            response = requests.post(url, headers=headers, json=data)

        accumulated_text = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    result = self._process_sse_line(line.decode('utf-8'))
                    if result is not None:
                        # sample payload for reference
                        # {'type': 'other', 'data': {'id': '52745409-b9c9-4033-a41d-17bcced7c11a', 'model': 'claude-3-5-sonnet', 
                        # 'choices': [{'delta': {'type': 'text', 'content': 'd significantly more support tickets than Business Internet services,', 
                        # 'content_list': [{'type': 'text', 'text': 'd significantly more support tickets than Business Internet services,'}], 
                        # 'text': 'd significantly more support tickets than Business Internet services,'}}], 'usage': {}}}
                        if result.get('type') == 'other':
                            if 'content' in result['data']['choices'][0]['delta']:
                                accumulated_text += result['data']['choices'][0]['delta']['content']
            return {'type': 'text', 'text': f'{accumulated_text}'}
        else:
            print(f"Error: Received status code {response.status_code} with message {response.json()}")
            return {'type': 'error', 'text': f'Error: Received status code {response.status_code} with message {response.json()}'}

    def _retrieve_response(self, query: str, limit=1) -> dict[str, any]:
        url = self.agent_url
        headers = {
            'X-Snowflake-Authorization-Token-Type': 'KEYPAIR_JWT',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f"Bearer {self.jwt}"
        }
        data = {
            "model": self.model,
            "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }
            ],
            "tools": [
                {
                    "tool_spec": {
                        "type": "cortex_search",
                        "name": "vehicles_info_search"
                    }
                },
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "supply_chain"
                    }
                },
                {
                    "tool_spec": {
                        "type": "cortex_analyst_text_to_sql",
                        "name": "support"
                    }
                }
            ],
            "tool_resources": {
                "vehicles_info_search": {
                    "name": self.search_service,
                    "max_results": limit,
                    "title_column": "title",
                    "id_column": "relative_path",
                },
                "supply_chain": {
                    "semantic_model_file": self.supply_chain_semantic_model
                },
                "support": {
                    "semantic_model_file": self.support_semantic_model
                }
            },
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 401:  # Unauthorized - likely expired JWT
            print("JWT has expired. Generating new JWT...")
            # Generate new token
            self.jwt = JWTGenerator(self.account, self.user, self.private_key_path).get_token()
            # Retry the request with the new token
            headers["Authorization"] = f"Bearer {self.jwt}"
            print("New JWT generated. Sending new request to Cortex Agents API. Please wait...")
            response = requests.post(url, headers=headers, json=data)

        if DEBUG:
            print(response.text)
        if response.status_code == 200:
            return self._parse_response(response)
        else:
            print(f"Error: Received status code {response.status_code} with message {response.json()}")
            return {'type': 'error', 'text': f'Error: Received status code {response.status_code} with message {response.json()}'}

    def _parse_delta_content(self,content: list) -> dict[str, any]:
        """Parse different types of content from the delta."""
        result = {
            'text': '',
            'tool_use': [],
            'tool_results': []
        }
        
        for entry in content:
            entry_type = entry.get('type')
            if entry_type == 'text':
                result['text'] += entry.get('text', '')
            elif entry_type == 'tool_use':
                result['tool_use'].append(entry.get('tool_use', {}))
            elif entry_type == 'tool_results':
                result['tool_results'].append(entry.get('tool_results', {}))
        
        return result

    def _process_sse_line(self,line: str) -> dict[str, any]:
        """Process a single SSE line and return parsed content."""
        if not line.startswith('data: '):
            return {}
        try:
            json_str = line[6:].strip()  # Remove 'data: ' prefix
            if json_str == '[DONE]':
                return {'type': 'done'}
                
            data = json.loads(json_str)
            if data.get('object') == 'message.delta':
                delta = data.get('delta', {})
                if 'content' in delta:
                    return {
                        'type': 'message',
                        'content': self._parse_delta_content(delta['content'])
                    }
            return {'type': 'other', 'data': data}
        except json.JSONDecodeError:
            return {'type': 'error', 'text': f'Failed to parse: {line}'}
    
    def _parse_response(self,response: requests.Response) -> dict[str, any]:
        """Parse and print the SSE chat response with improved organization."""
        accumulated = {
            'text': '',
            'tool_use': [],
            'tool_results': [],
            'other': []
        }

        for line in response.iter_lines():
            if line:
                result = self._process_sse_line(line.decode('utf-8'))
                
                if result.get('type') == 'message':
                    content = result['content']
                    accumulated['text'] += content['text']
                    accumulated['tool_use'].extend(content['tool_use'])
                    accumulated['tool_results'].extend(content['tool_results'])
                elif result.get('type') == 'other':
                    accumulated['other'].append(result['data'])

        text = ''
        sql = ''
        citations = ''

        if accumulated['text']:
            text = accumulated['text']

        if DEBUG:
            print("\n=== Complete Response ===")

            print("\n--- Generated Text ---")
            print(text)

            if accumulated['tool_use']:
                print("\n--- Tool Usage ---")
                print(json.dumps(accumulated['tool_use'], indent=2))

            if accumulated['other']:
                print("\n--- Other Messages ---")
                print(json.dumps(accumulated['other'], indent=2))

            if accumulated['tool_results']:
                print("\n--- Tool Results ---")
                print(json.dumps(accumulated['tool_results'], indent=2))

        if accumulated['tool_results']:
            for result in accumulated['tool_results']:
                for k,v in result.items():
                    if k == 'content':
                        for content in v:
                            if 'sql' in content['json']:
                                text = content['json']['text'] 
                                sql = content['json']['sql']
                            elif 'searchResults' in content['json']:
                                search_results = content['json']['searchResults']
                                for search_result in search_results:
                                    citations += f"{search_result['text']}"
                                text = text.replace("【†1†】","").replace("【†2†】","").replace("【†3†】","").replace(" .",".") + "*"
                                citations = f"{search_result['doc_title']} \n {citations} \n\n[Source: {search_result['doc_id']}]"

        return {"text": text, "sql": sql, "citations": citations}
       
    def chat(self, query: str) -> any:
        response = self._retrieve_response(query)
        return response
