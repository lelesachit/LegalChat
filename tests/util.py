import logging
import argparse

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level= logging.INFO,
                    handlers=[
                        logging.FileHandler('legalchat.log'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger('legalchat')

def get_parser()->argparse.ArgumentParser: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9500)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--pretrained', type=str, default ='./quant_llm/llama-2-7b-chat.Q4_K_S.gguf' )

    return parser

