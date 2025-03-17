import os
from typing import Dict, Any, Tuple, Optional
import litellm
import requests
import time
import tiktoken

class LLMService:
    def __init__(self, api_key: Optional[str] = None):
        # HuggingFace API token (can be empty for some public models)
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Google API key for Gemini
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Warn if the Google API key is not set
        if not self.google_api_key:
            print("Warning: GOOGLE_API_KEY environment variable is not set. Gemini API calls will fail.")
        
        # Define model pricing (per 1M tokens) in USD - based on provided pricing
        self.model_pricing = {
            "huggingface/HuggingFaceH4/zephyr-7b-beta": {
                "input": 0.0,
                "output": 0.0
            },
            "gemini/gemini-pro": {
                "input": {
                    "standard": 1.25,  # $1.25 per 1M input tokens (≤128k)
                    "high": 2.50       # $2.50 per 1M input tokens (>128k)
                },
                "output": {
                    "standard": 5.00,  # $5.00 per 1M output tokens (≤128k)
                    "high": 10.00      # $10.00 per 1M output tokens (>128k)
                },
                "threshold": 128000    # Token threshold for pricing tier
            }
        }
        
        # Initialize tiktoken encoder for token counting
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def get_available_models(self) -> list:
        """Return list of available models"""
        models = [
            {
                "id": "huggingface/HuggingFaceH4/zephyr-7b-beta",
                "name": "Zephyr 7B",
                "provider": "HuggingFace via LiteLLM"
            }
        ]
        
        # Add Gemini model if API key is available
        if self.google_api_key:
            models.append({
                "id": "gemini/gemini-pro",
                "name": "Google Gemini Pro",
                "provider": "Google via LiteLLM"
            })
            
        return models
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def _chunk_document(self, document: str, max_chunk_size: int = 4000) -> list:
        """Split document into manageable chunks."""
        # Split by paragraphs first
        paragraphs = document.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, start a new chunk
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
            
    def generate_summary(self, document_content: str, model_id: str = "huggingface/HuggingFaceH4/zephyr-7b-beta") -> Tuple[str, Dict[str, Any]]:
        """
        Generate a summary of the document using the specified model.
        
        Args:
            document_content: The text content of the document.
            model_id: The ID of the model to use.
            
        Returns:
            Tuple containing the summary and cost information.
        """
        system_prompt = "You are a helpful assistant that summarizes documents. Provide a concise but comprehensive summary of the document."
        
        # Check if document is too long and we're using Zephyr
        if self._count_tokens(document_content) > 6000 and model_id.startswith("huggingface"):
            # Chunk the document
            chunks = self._chunk_document(document_content)
            chunk_summaries = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            print(f"Document chunked into {len(chunks)} parts for processing")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"Please summarize the following part {i+1} of {len(chunks)} of the document:\n\n{chunk}"
                
                chunk_summary = self._call_huggingface_api(system_prompt, chunk_prompt)
                chunk_summaries.append(chunk_summary)
                
                # Count tokens
                prompt_tokens = self._count_tokens(system_prompt + chunk_prompt)
                completion_tokens = self._count_tokens(chunk_summary)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                print(f"Processed chunk {i+1}/{len(chunks)}")
            
            # Combine chunk summaries
            combined_summary = "\n\n".join(chunk_summaries)
            
            # Generate a final summary of the summaries
            final_prompt = f"Please create a cohesive final summary from these section summaries:\n\n{combined_summary}"
            
            final_summary = self._call_huggingface_api(system_prompt, final_prompt)
            
            # Add token counts
            final_prompt_tokens = self._count_tokens(system_prompt + final_prompt)
            final_completion_tokens = self._count_tokens(final_summary)
            total_prompt_tokens += final_prompt_tokens
            total_completion_tokens += final_completion_tokens
            
            cost_info = self._calculate_cost(model_id, total_prompt_tokens, total_completion_tokens)
            
            return final_summary, cost_info
        else:
            # Original implementation for shorter documents or Gemini
            user_prompt = f"Please summarize the following document:\n\n{document_content}"
            
            try:
                if model_id.startswith("gemini"):
                    summary = self._call_gemini_api(system_prompt, user_prompt)
                else:
                    summary = self._call_huggingface_api(system_prompt, user_prompt)
                
                # Count tokens with tiktoken
                prompt_tokens = self._count_tokens(system_prompt + user_prompt)
                completion_tokens = self._count_tokens(summary)
                
                cost_info = self._calculate_cost(model_id, prompt_tokens, completion_tokens)
                
                return summary, cost_info
            except Exception as e:
                print(f"Error calling LLM API: {str(e)}")
                return "Unable to generate summary due to an error.", {
                    "model": model_id,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_cost": 0,
                    "output_cost": 0,
                    "total_cost": 0
                }
    
    def answer_question(self, document_content: str, question: str, model_id: str = "huggingface/HuggingFaceH4/zephyr-7b-beta") -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question about the document provided by the user using the specified model.
        
        Args:
            document_content: The text content of the document.
            question: The question to answer.
            model_id: The ID of the model to use.
            
        Returns:
            Tuple containing the answer and cost information.
        """
        system_prompt = "You are a helpful assistant that answers questions only restricted to the document content provided. Provide accurate and concise answers based on the document content only."
        
        # Check if document is too long and we're using Zephyr
        if self._count_tokens(document_content) > 6000 and model_id.startswith("huggingface"):
            # Chunk the document
            chunks = self._chunk_document(document_content)
            chunk_answers = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            print(f"Document chunked into {len(chunks)} parts for QA processing")
            
            # Process each chunk to find potential answers
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"Document part {i+1} of {len(chunks)}:\n\n{chunk}\n\nQuestion: {question}\n\nIf you can answer the question based on this document part, provide the answer. If not, respond with 'No relevant information in this part.'"
                
                chunk_answer = self._call_huggingface_api(system_prompt, chunk_prompt)
                
                # Only keep relevant answers
                if "No relevant information in this part" not in chunk_answer:
                    chunk_answers.append(chunk_answer)
                
                # Count tokens
                prompt_tokens = self._count_tokens(system_prompt + chunk_prompt)
                completion_tokens = self._count_tokens(chunk_answer)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                print(f"Processed QA chunk {i+1}/{len(chunks)}")
            
            # If we found relevant answers
            if chunk_answers:
                # Combine relevant answers
                combined_answers = "\n\n".join(chunk_answers)
                
                # Generate a final consolidated answer
                final_prompt = f"I found these potential answers to the question '{question}':\n\n{combined_answers}\n\nPlease provide a single coherent answer based on these findings."
                
                final_answer = self._call_huggingface_api(system_prompt, final_prompt)
                
                # Add token counts
                final_prompt_tokens = self._count_tokens(system_prompt + final_prompt)
                final_completion_tokens = self._count_tokens(final_answer)
                total_prompt_tokens += final_prompt_tokens
                total_completion_tokens += final_completion_tokens
                
                cost_info = self._calculate_cost(model_id, total_prompt_tokens, total_completion_tokens)
                
                return final_answer, cost_info
            else:
                # No relevant information found in any chunk
                answer = "I cannot find information about this in the document."
                cost_info = self._calculate_cost(model_id, total_prompt_tokens, total_completion_tokens)
                return answer, cost_info
        else:
            # Original implementation for shorter documents or Gemini
            user_prompt = f"Document: {document_content}\n\nQuestion: {question}\n\nAnswer:"
            
            try:
                if model_id.startswith("gemini"):
                    answer = self._call_gemini_api(system_prompt, user_prompt)
                else:
                    answer = self._call_huggingface_api(system_prompt, user_prompt)
                
                # Count tokens with tiktoken
                prompt_tokens = self._count_tokens(system_prompt + user_prompt)
                completion_tokens = self._count_tokens(answer)
                
                cost_info = self._calculate_cost(model_id, prompt_tokens, completion_tokens)
                
                return answer, cost_info
            except Exception as e:
                print(f"Error calling LLM API: {str(e)}")
                return "Unable to answer question due to an error.", {
                    "model": model_id,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_cost": 0,
                    "output_cost": 0,
                    "total_cost": 0
                }
    
    def _call_huggingface_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the HuggingFace API with improved prompting."""
        try:
            # Determine if this is a summary or QA task
            is_summary = "summarize" in user_prompt.lower()
            
            if is_summary:
                formatted_prompt = f"""
                {system_prompt}
                
                I need you to create a comprehensive summary of the following document. 
                Focus on the main topics, key points, and important details.
                Organize the summary with clear sections and bullet points where appropriate.
                
                Document:
                {user_prompt.replace('Please summarize the following document:', '').strip()}
                
                Summary:
                """
            else:
                # This is a QA task
                # Extract the question from the user prompt
                if "Question:" in user_prompt:
                    document_part = user_prompt.split("Question:")[0].strip()
                    question_part = user_prompt.split("Question:")[1].split("Answer:")[0].strip() if "Answer:" in user_prompt else user_prompt.split("Question:")[1].strip()
                else:
                    document_part = user_prompt
                    question_part = ""
                
                formatted_prompt = f"""
                {system_prompt}
                
                I need you to answer the following question based ONLY on the information provided in the document.
                If the answer is not in the document, say "I cannot find information about this in the document."
                
                Document:
                {document_part}
                
                Question:
                {question_part}
                
                Answer:
                """
            
            # Use the HuggingFace Inference API endpoint
            api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            
            # Improve parameters for better output
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 800,  # Increased for more comprehensive responses
                    "temperature": 0.2,  # Lower temperature for more focused output
                    "do_sample": True,
                    "top_p": 0.85,  # Slightly adjusted top_p
                    "top_k": 40,  # Add top_k filtering
                    "repetition_penalty": 1.2,  # Penalize repetition
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    # Clean up the response
                    if "Summary:" in generated_text and is_summary:
                        generated_text = generated_text.split("Summary:")[1].strip()
                    elif "Answer:" in generated_text and not is_summary:
                        generated_text = generated_text.split("Answer:")[1].strip()
                    return generated_text
                return "No valid response from HuggingFace model."
            else:
                # Add retry logic for 503 errors (model loading)
                if response.status_code == 503:
                    print("Model is loading, retrying in 5 seconds...")
                    time.sleep(5)
                    return self._call_huggingface_api(system_prompt, user_prompt)
                return f"Error: API returned status code {response.status_code}"
                
        except Exception as e:
            print(f"Exception calling HuggingFace API: {str(e)}")
            return f"Error: {str(e)}"
        
    def _call_gemini_api(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Google Gemini API using LiteLLM."""
        try:
            # Set the API key as an environment variable
            os.environ['GEMINI_API_KEY'] = self.google_api_key
            
            # Print debug info (remove in production)
            print(f"Using Gemini API with key: {self.google_api_key[:5]}...{self.google_api_key[-4:] if len(self.google_api_key) > 8 else ''}")
            
            # Ensure a valid Google API key is provided
            if not self.google_api_key:
                raise ValueError("Google API key is not provided. Set the GOOGLE_API_KEY environment variable with a valid key.")
            
            # Create messages for the API call
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call the Gemini API
            response = litellm.completion(
                model="gemini/gemini-1.5-pro",  # Using the 1.5 version
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            
            # Extract the generated text if available
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            
            return "No valid response from Gemini model."
        except Exception as e:
            print(f"Exception calling Gemini API: {str(e)}")
            return f"Error: {str(e)}"
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Calculate the cost of the API call based on token usage."""
        if model_id.startswith("gemini"):
            # Get the pricing tiers
            pricing = self.model_pricing.get("gemini/gemini-pro", {})
            threshold = pricing.get("threshold", 128000)
            
            # Determine which pricing tier to use
            input_tier = "high" if input_tokens > threshold else "standard"
            output_tier = "high" if output_tokens > threshold else "standard"
            
            # Calculate costs
            input_cost = (input_tokens / 1000000) * pricing["input"][input_tier]
            output_cost = (output_tokens / 1000000) * pricing["output"][output_tier]
        else:
            # For other models
            model_price = self.model_pricing.get(model_id, {"input": 0, "output": 0})
            input_cost = (input_tokens / 1000) * model_price["input"]
            output_cost = (output_tokens / 1000) * model_price["output"]
            
        total_cost = input_cost + output_cost
        
        return {
            "model": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }