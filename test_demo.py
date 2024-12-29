import torch

from swap import text_to_tensor
from train import device, model
from vocalbulary import vocab
from wash import preprocess_text


def generate_response(model, input_text, max_len=50):
    model.eval()
    tokens = preprocess_text(input_text)
    src_tensor = text_to_tensor(tokens, vocab).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_output, hidden = model.encoder(src_tensor)
        trg_token = torch.tensor([vocab['<sos>']]).unsqueeze(0).to(device)
        response = []

        for _ in range(max_len):
            output = model.decoder(trg_token, hidden, enc_output)
            next_token = output.argmax(dim=-1)[-1].item()
            if next_token == vocab['<eos>']:
                break
            response.append(next_token)
            trg_token = torch.tensor([next_token]).unsqueeze(0).to(device)

    response_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in response]
    return ' '.join(response_tokens)


# 测试生成对话
input_text = "Hello, how are you?"
response = generate_response(model, input_text)
print(f"Response: {response}")
