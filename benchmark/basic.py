import sys
import aiohttp
import asyncio


async def query_model_vllm(prompt, output_len=4096):
    ip_port = 'localhost:8000'
    timeout = aiohttp.ClientTimeout(total=4 * 60 * 60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        request_dict = {
            "prompt": prompt,
            "n": 1,
            "best_of": 1,
            "temperature": 1.0,
            "top_k": 1,
            "max_tokens": max(output_len, 1),
            "ignore_eos": True,
            "stream": False,
        }
        print('Querying model')
        try:
            async with session.post(f'http://{ip_port}/generate_benchmark', json=request_dict) as resp:
                print('post done')
                output = await resp.json()
                print(output)
                # if 'generated_text' in output:
                #     print(json.dumps(output['generated_text']))
        except aiohttp.ClientError as e:
            print(f"Connect to {ip_port} failed with: {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    asyncio.run(query_model_vllm("Hello, my name is"))
