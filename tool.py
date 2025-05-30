import pygame
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk

class AnnotationTool:
    def __init__(self):
        pygame.init()
        
        # Configurações da tela
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("TwinLiteNet Dataset Annotation Tool")
        
        # Cores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Estado da aplicação
        self.current_mode = "line"  # "line" ou "polygon"
        self.images_folder = None
        self.output_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.original_image = None
        self.image_scale = 1.0
        self.image_offset = [0, 0]
        
        # Desenho
        self.drawing = False
        self.current_line = []
        self.lines = []
        self.current_polygon = []
        self.polygons = []
        self.line_thickness = 3
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Criar pastas de output
        self.setup_output_folders()
        
    def setup_output_folders(self):
        """Cria as pastas de output se não existirem"""
        self.lines_folder = Path("output/lines")
        self.segmentation_folder = Path("output/segmentation")
        
        self.lines_folder.mkdir(parents=True, exist_ok=True)
        self.segmentation_folder.mkdir(parents=True, exist_ok=True)
        
    def select_folder(self):
        """Seleciona a pasta com as imagens"""
        root = tk.Tk()
        root.withdraw()
        
        folder = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if folder:
            self.images_folder = Path(folder)
            self.load_images()
            
        root.destroy()
        
    def load_images(self):
        """Carrega todas as imagens da pasta selecionada"""
        if not self.images_folder:
            return
            
        # Extensões de imagem suportadas
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        self.image_files = []
        for ext in extensions:
            self.image_files.extend(self.images_folder.glob(f"*{ext}"))
            self.image_files.extend(self.images_folder.glob(f"*{ext.upper()}"))
            
        self.image_files.sort()
        
        if self.image_files:
            self.current_image_index = 0
            self.load_current_image()
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada na pasta selecionada!")
            
    def load_current_image(self):
        """Carrega a imagem atual"""
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        image_path = self.image_files[self.current_image_index]
        
        try:
            # Carrega a imagem usando OpenCV
            cv_image = cv2.imread(str(image_path))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Converte para pygame surface
            self.original_image = pygame.surfarray.make_surface(cv_image.swapaxes(0, 1))
            
            # Redimensiona para caber na tela
            self.fit_image_to_screen()
            
            # Limpa as anotações
            self.clear_annotations()
            
        except Exception as e:
            print(f"Erro ao carregar imagem {image_path}: {e}")
            
    def fit_image_to_screen(self):

        if not self.original_image:
            return
            

        available_width = self.SCREEN_WIDTH - 200
        available_height = self.SCREEN_HEIGHT - 100
        
        img_width = self.original_image.get_width()
        img_height = self.original_image.get_height()
        

        scale_x = available_width / img_width
        scale_y = available_height / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)  # Máximo 1.0 para não aumentar
        
        # Redimensiona a imagem
        new_width = int(img_width * self.image_scale)
        new_height = int(img_height * self.image_scale)
        self.current_image = pygame.transform.scale(self.original_image, (new_width, new_height))
        
        # Centraliza a imagem
        self.image_offset = [
            (available_width - new_width) // 2,
            (available_height - new_height) // 2 + 50
        ]
        
    def clear_annotations(self):
        """Limpa todas as anotações"""
        self.lines = []
        self.polygons = []
        self.current_line = []
        self.current_polygon = []
        self.drawing = False
        
    def get_image_coords(self, screen_pos):
        if not self.current_image:
            return None
            
        x, y = screen_pos
        
        # Remove o offset
        img_x = x - self.image_offset[0]
        img_y = y - self.image_offset[1]
        
        # Verifica se está dentro da imagem
        if (img_x < 0 or img_y < 0 or 
            img_x >= self.current_image.get_width() or 
            img_y >= self.current_image.get_height()):
            return None
            
        # Converte para coordenadas da imagem original
        orig_x = int(img_x / self.image_scale)
        orig_y = int(img_y / self.image_scale)
        
        return (orig_x, orig_y)
    
    def get_screen_coords(self, image_pos):
        orig_x, orig_y = image_pos
        

        img_x = int(orig_x * self.image_scale)
        img_y = int(orig_y * self.image_scale)
        

        screen_x = img_x + self.image_offset[0]
        screen_y = img_y + self.image_offset[1]
        
        return (screen_x, screen_y)
        
    def handle_mouse_click(self, pos, button):
        if not self.current_image:
            return
            
        image_pos = self.get_image_coords(pos)
        if not image_pos:
            return
            
        if self.current_mode == "line":
            if button == 1:  # Botão esquerdo
                if not self.drawing:
                    # Inicia nova linha
                    self.current_line = [image_pos]
                    self.drawing = True
                else:
                    # Adiciona ponto à linha atual
                    self.current_line.append(image_pos)
            elif button == 3 and self.drawing:  # Botão direito - finaliza linha
                if len(self.current_line) >= 2:
                    self.lines.append(self.current_line.copy())
                self.current_line = []
                self.drawing = False
                
        elif self.current_mode == "polygon":
            if button == 1:  # Botão esquerdo
                self.current_polygon.append(image_pos)
            elif button == 3:  # Botão direito - finaliza polígono
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                self.current_polygon = []
                
    def handle_mouse_motion(self, pos):
        """Manipula movimento do mouse"""
        if not self.current_image or not self.drawing:
            return
            
        if self.current_mode == "line" and self.drawing:
            image_pos = self.get_image_coords(pos)
            if image_pos:
                # Atualiza o último ponto da linha atual
                if len(self.current_line) > 0:
                    # Remove o último ponto temporário se existir
                    if len(self.current_line) > 1:
                        self.current_line = self.current_line[:-1]
                    self.current_line.append(image_pos)
                    
    def draw_annotations(self):
        """Desenha todas as anotações na tela"""
        if not self.current_image:
            return
            

        for line in self.lines:
            if len(line) >= 2:
                screen_points = [self.get_screen_coords(p) for p in line]
                pygame.draw.lines(self.screen, self.GREEN, False, screen_points, self.line_thickness)
                

        if len(self.current_line) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_line]
            pygame.draw.lines(self.screen, self.YELLOW, False, screen_points, self.line_thickness)
            

        for polygon in self.polygons:
            if len(polygon) >= 3:
                screen_points = [self.get_screen_coords(p) for p in polygon]
                pygame.draw.polygon(self.screen, self.BLUE, screen_points)
                pygame.draw.polygon(self.screen, self.RED, screen_points, 2)
                

        if len(self.current_polygon) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_polygon]
            pygame.draw.lines(self.screen, self.YELLOW, False, screen_points, 2)
            
     
        for point in self.current_polygon:
            screen_point = self.get_screen_coords(point)
            pygame.draw.circle(self.screen, self.YELLOW, screen_point, 3)
            
    def save_annotations(self):
        if not self.current_image or not self.image_files:
            return
            
        current_file = self.image_files[self.current_image_index]
        filename = current_file.stem
        

        orig_width = self.original_image.get_width()
        orig_height = self.original_image.get_height()
        

        lines_image = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        # Desenha linhas
        for line in self.lines:
            if len(line) >= 2:
                points = np.array(line, dtype=np.int32)
                cv2.polylines(lines_image, [points], False, 255, self.line_thickness)
                
        # Cria imagem para segmentação (fundo preto, polígonos brancos)
        seg_image = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        # Desenha polígonos
        for polygon in self.polygons:
            if len(polygon) >= 3:
                points = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(seg_image, [points], 255)
                

        try:
            cv2.imwrite(str(self.lines_folder / f"{filename}.png"), lines_image)
            cv2.imwrite(str(self.segmentation_folder / f"{filename}.png"), seg_image)
            print(f"Anotações guardar para {filename}")
        except Exception as e:
            print(f"Erro ao guardar anotações: {e}")
            
    def draw_ui(self):
        panel_rect = pygame.Rect(self.SCREEN_WIDTH - 180, 0, 180, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        
        y_pos = 20
        

        if self.image_files:
            text = f"Imagem: {self.current_image_index + 1}/{len(self.image_files)}"
            text_surface = self.small_font.render(text, True, self.WHITE)
            self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
            y_pos += 30
            
            filename = self.image_files[self.current_image_index].name
            if len(filename) > 20:
                filename = filename[:17] + "..."
            text_surface = self.small_font.render(filename, True, self.WHITE)
            self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
            y_pos += 40
            

        mode_text = f"Modo: {self.current_mode.upper()}"
        mode_color = self.GREEN if self.current_mode == "line" else self.BLUE
        text_surface = self.font.render(mode_text, True, mode_color)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
        y_pos += 40
        
        # Estatísticas
        text_surface = self.small_font.render(f"Linhas: {len(self.lines)}", True, self.WHITE)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
        y_pos += 25
        
        text_surface = self.small_font.render(f"Polígonos: {len(self.polygons)}", True, self.WHITE)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
        y_pos += 40
        

        instructions = [
            "CONTROLES:",
            "",
            "F - Selecionar pasta",
            "SPACE - Trocar modo",
            "S - Salvar anotações",
            "C - Limpar tudo",
            "A/D - Imagem ant/próx",
            "",
            "LINHA:",
            "Click esq - Adicionar ponto",
            "Click dir - Finalizar",
            "",
            "POLÍGONO:",
            "Click esq - Adicionar ponto",
            "Click dir - Finalizar",
            "",
            "ESC - Sair"
        ]
        
        for instruction in instructions:
            text_surface = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(text_surface, (self.SCREEN_WIDTH - 170, y_pos))
            y_pos += 20
            
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def prev_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        self.select_folder()
                    elif event.key == pygame.K_SPACE:
                        self.current_mode = "polygon" if self.current_mode == "line" else "line"
                        self.drawing = False
                        self.current_line = []
                        self.current_polygon = []
                    elif event.key == pygame.K_s:
                        self.save_annotations()
                    elif event.key == pygame.K_c:
                        self.clear_annotations()
                    elif event.key == pygame.K_a:
                        self.prev_image()
                    elif event.key == pygame.K_d:
                        self.next_image()
                        
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event.pos, event.button)
                    
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)
                    

            self.screen.fill(self.BLACK)
            

            if self.current_image:
                self.screen.blit(self.current_image, self.image_offset)
                

            self.draw_annotations()
            

            self.draw_ui()
            
            pygame.display.flip()
            clock.tick(60)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    tool = AnnotationTool()
    tool.run()
