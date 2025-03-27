import torch

# Verifica se o PyTorch está instalado
print("PyTorch versão:", torch.__version__)

# Testa se a GPU está disponível
gpu_disponivel = torch.cuda.is_available()
print("GPU disponível:", gpu_disponivel)

if gpu_disponivel:
    # Mostra informações da GPU
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Capacidade da GPU:", torch.cuda.get_device_capability(0))
    
    # Testa operações simples na GPU
    tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("Tensor na GPU:", tensor)
else:
    print("PyTorch não está utilizando a GPU. Verifique a instalação.")

