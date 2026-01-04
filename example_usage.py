#!/usr/bin/env python3
"""
Example script demonstrating the Compliance Auditor API usage
"""

import requests
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_chat_completion(query: str, stream: bool = False):
    """Test the chat completion endpoint"""
    print("=" * 60)
    print(f"Testing Chat Completion (Stream={stream})")
    print("=" * 60)
    print(f"Query: {query}")
    print("-" * 60)
    
    payload = {
        "model": "llama2",
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": stream,
        "temperature": 0.7
    }
    
    if stream:
        # Streaming response
        with requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload,
            stream=True
        ) as response:
            print(f"Status Code: {response.status_code}")
            print("Streaming Response:")
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str != '[DONE]':
                            try:
                                chunk = json.loads(data_str)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        print(delta['content'], end='', flush=True)
                            except json.JSONDecodeError:
                                pass
            print("\n")
    else:
        # Non-streaming response
        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions",
            json=payload
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Model: {result.get('model')}")
            print(f"Response ID: {result.get('id')}")
            
            if result.get('choices'):
                message = result['choices'][0]['message']['content']
                print("\nResponse:")
                print("-" * 60)
                print(message)
        else:
            print(f"Error: {response.text}")
    
    print()

def main():
    """Main example execution"""
    print("\n" + "=" * 60)
    print("Compliance Auditor API - Example Usage")
    print("=" * 60 + "\n")
    
    try:
        # Test health endpoint
        test_health()
        
        # Example queries
        queries = [
            "Was sind die DORA-Anforderungen für IKT-Risikomanagement?",
            "Welche Compliance-Anforderungen gibt es für Datensicherheit?",
            "Wie ist das Incident-Management geregelt?"
        ]
        
        # Test non-streaming
        for query in queries[:1]:  # Test with first query
            test_chat_completion(query, stream=False)
        
        # Test streaming (uncomment to test)
        # test_chat_completion(queries[0], stream=True)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API.")
        print("   Make sure the server is running:")
        print("   python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
