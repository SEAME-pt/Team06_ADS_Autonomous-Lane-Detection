import os
import shutil
from pathlib import Path

def organize_dataset_separate_folders(source_dir="dataset", target_dir="organized_dataset", train_ratio=0.8):
    """
    Organiza o dataset em pastas separadas para images, segments e lanes
    
    Estrutura resultante:
    organized_dataset/
    ├── train/
    │   ├── images/
    │   ├── segments/
    │   └── lanes/
    └── val/
        ├── images/
        ├── segments/
        └── lanes/
    """
    
    # Limpar diretório de destino se existir
    target_path = Path(target_dir)
    if target_path.exists():
        shutil.rmtree(target_path)
        print(f"Limpando diretório existente: {target_path}")
    
    # Criar diretório de destino
    target_path.mkdir(parents=True)
    
    # Caminhos das pastas de origem
    images_dir = Path(source_dir) / "images"
    lines_dir = Path(source_dir) / "lines" 
    segmentation_dir = Path(source_dir) / "segmentation"
    
    # Verificar se as pastas existem
    if not all([images_dir.exists(), lines_dir.exists(), segmentation_dir.exists()]):
        print("Erro: Certifique-se que existem as pastas 'images', 'lines' e 'segmentation'")
        print(f"Verificando:")
        print(f"  {images_dir} - {'✓' if images_dir.exists() else '✗'}")
        print(f"  {lines_dir} - {'✓' if lines_dir.exists() else '✗'}")
        print(f"  {segmentation_dir} - {'✓' if segmentation_dir.exists() else '✗'}")
        return
    
    # Criar estrutura de destino
    train_dir = target_path / "train"
    val_dir = target_path / "val"
    
    # Criar subdiretórios
    for split_dir in [train_dir, val_dir]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "segments").mkdir(parents=True, exist_ok=True)
        (split_dir / "lanes").mkdir(parents=True, exist_ok=True)
    
    # Obter lista de imagens
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print("Nenhuma imagem encontrada na pasta images!")
        return
    
    print(f"Encontradas {len(image_files)} imagens")
    
    # Calcular split
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    
    # Processar cada imagem
    successful_copies = 0
    missing_files = []
    
    for i, img_path in enumerate(image_files):
        img_name = img_path.stem  # nome sem extensão
        img_ext = img_path.suffix  # .jpg ou .png
        
        # Determinar se vai para treino ou validação
        is_train = i < train_count
        dest_base_dir = train_dir if is_train else val_dir
        
        # Caminhos dos arquivos correspondentes
        line_file = lines_dir / f"{img_name}.png"
        seg_file = segmentation_dir / f"{img_name}.png"
        
        # Verificar se existem os arquivos correspondentes
        files_exist = True
        if not line_file.exists():
            missing_files.append(f"LINE: {img_name}")
            files_exist = False
            
        if not seg_file.exists():
            missing_files.append(f"SEG: {img_name}")
            files_exist = False
        
        if not files_exist:
            continue
        
        try:
            # Copiar para as pastas apropriadas
            # Imagem
            shutil.copy2(img_path, dest_base_dir / "images" / f"{img_name}{img_ext}")
            
            # Segmentação
            shutil.copy2(seg_file, dest_base_dir / "segments" / f"{img_name}.png")
            
            # Linhas
            shutil.copy2(line_file, dest_base_dir / "lanes" / f"{img_name}.png")
            
            successful_copies += 1
            
            if successful_copies % 100 == 0:
                print(f"Processadas {successful_copies} imagens...")
                
        except Exception as e:
            print(f"Erro ao processar {img_name}: {e}")
            continue
    
    print(f"\n=== Organização concluída! ===")
    print(f"Imagens processadas com sucesso: {successful_copies}")
    
    # Contar arquivos em cada split
    train_images = len(list((train_dir / "images").glob("*")))
    val_images = len(list((val_dir / "images").glob("*")))
    
    print(f"Treino: {train_images} conjuntos completos")
    print(f"Validação: {val_images} conjuntos completos")
    
    if missing_files:
        print(f"\nArquivos faltando ({len(missing_files)}):")
        for missing in missing_files[:10]:  # mostrar apenas os primeiros 10
            print(f"  {missing}")
        if len(missing_files) > 10:
            print(f"  ... e mais {len(missing_files) - 10}")
    
    print(f"\nDataset organizado em: {target_path.absolute()}")
    return target_path



 
organize_dataset_separate_folders(
        source_dir="dataset",
        target_dir="organized_dataset", 
        train_ratio=0.8
    )
    
    
