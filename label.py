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
        

        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(" Dataset Annotation Tool Team06")
        
        # Cores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Estado da aplicação
        self.current_mode = "line"  # "line", "polygon" ou "edit"
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
        
        # Edição
        self.selected_point = None  
        self.dragging = False
        self.selection_radius = 10  # Raio para seleção de pontos
        
        # UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        

        self.setup_output_folders()
    
    def setup_output_folders(self):
        self.lines_folder = Path("output/lines")
        self.segmentation_folder = Path("output/segmentation")
        self.lines_folder.mkdir(parents=True, exist_ok=True)
        self.segmentation_folder.mkdir(parents=True, exist_ok=True)
    
    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Seleciona a pasta com as imagens")
        if folder:
            self.images_folder = Path(folder)
            self.load_images()
        root.destroy()
    
    def load_images(self):
        if not self.images_folder:
            return
        

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
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_image_index]
        try:

            cv_image = cv2.imread(str(image_path))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            

            self.original_image = pygame.surfarray.make_surface(cv_image.swapaxes(0, 1))
            

            self.fit_image_to_screen()
            

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

        self.lines = []
        self.polygons = []
        self.current_line = []
        self.current_polygon = []
        self.drawing = False
        self.selected_point = None
        self.dragging = False
    
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
        
        #  escala
        img_x = int(orig_x * self.image_scale)
        img_y = int(orig_y * self.image_scale)
        
        #  offset
        screen_x = img_x + self.image_offset[0]
        screen_y = img_y + self.image_offset[1]
        
        return (screen_x, screen_y)
    
    def find_nearest_point(self, pos):

        image_pos = self.get_image_coords(pos)
        if not image_pos:
            return None
        
        min_distance = float('inf')
        nearest_point = None
        

        for line_idx, line in enumerate(self.lines):
            for point_idx, point in enumerate(line):
                screen_point = self.get_screen_coords(point)
                distance = ((screen_point[0] - pos[0])**2 + (screen_point[1] - pos[1])**2)**0.5
                if distance < self.selection_radius and distance < min_distance:
                    min_distance = distance
                    nearest_point = ("line", line_idx, point_idx)
        

        for poly_idx, polygon in enumerate(self.polygons):
            for point_idx, point in enumerate(polygon):
                screen_point = self.get_screen_coords(point)
                distance = ((screen_point[0] - pos[0])**2 + (screen_point[1] - pos[1])**2)**0.5
                if distance < self.selection_radius and distance < min_distance:
                    min_distance = distance
                    nearest_point = ("polygon", poly_idx, point_idx)
        
        return nearest_point
    
    def delete_point(self, point_info):
        shape_type, shape_idx, point_idx = point_info
        
        if shape_type == "line":
            if len(self.lines[shape_idx]) > 2: 
                del self.lines[shape_idx][point_idx]
            else:
                del self.lines[shape_idx] 
        
        elif shape_type == "polygon":
            if len(self.polygons[shape_idx]) > 3: 
                del self.polygons[shape_idx][point_idx]
            else:
                del self.polygons[shape_idx]  
    
    def handle_mouse_click(self, pos, button):
        if not self.current_image:
            return

        image_pos = self.get_image_coords(pos)
        if not image_pos:
            return

        if self.current_mode == "edit":
            if button == 1:  # Botão esquerdo
                nearest_point = self.find_nearest_point(pos)
                if nearest_point:
                    self.selected_point = nearest_point
                    self.dragging = True
                else:
                    self.selected_point = None
            elif button == 3:  # Botão direito - delete ponto
                nearest_point = self.find_nearest_point(pos)
                if nearest_point:
                    self.delete_point(nearest_point)
        
        elif self.current_mode == "line":
            if button == 1:  # Botão esquerdo
                if not self.drawing:
                    self.current_line = [image_pos]
                    self.drawing = True
                else:
                    self.current_line.append(image_pos)
            elif button == 3 and self.drawing:  # Botão direito - finaliza linha
                if len(self.current_line) >= 2:
                    self.lines.append(self.current_line.copy())
                    self.current_line = []
                    self.drawing = False

        elif self.current_mode == "polygon":
            if button == 1:  # Botão esquerdo
                self.current_polygon.append(image_pos)
            elif button == 3:  # Botão direito - finaliza poly
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                    self.current_polygon = []
    
    def handle_mouse_up(self, pos, button):
        if button == 1 and self.dragging:
            self.dragging = False
            self.selected_point = None
    
    def handle_mouse_motion(self, pos):
        if not self.current_image:
            return
        
        if self.current_mode == "edit" and self.dragging and self.selected_point:
            image_pos = self.get_image_coords(pos)
            if image_pos:
                shape_type, shape_idx, point_idx = self.selected_point
                if shape_type == "line":
                    self.lines[shape_idx][point_idx] = image_pos
                elif shape_type == "polygon":
                    self.polygons[shape_idx][point_idx] = image_pos
        
        elif self.current_mode == "line" and self.drawing:
            image_pos = self.get_image_coords(pos)
            if image_pos:
                if len(self.current_line) > 0:
                    if len(self.current_line) > 1:
                        self.current_line = self.current_line[:-1]
                    self.current_line.append(image_pos)
    
    def draw_annotations(self):
        if not self.current_image:
            return


        for line in self.lines:
            if len(line) >= 2:
                screen_points = [self.get_screen_coords(p) for p in line]
                pygame.draw.lines(self.screen, self.GREEN, False, screen_points, self.line_thickness)
                

                if self.current_mode == "edit":
                    for point in line:
                        screen_point = self.get_screen_coords(point)
                        pygame.draw.circle(self.screen, self.WHITE, screen_point, 5)
                        pygame.draw.circle(self.screen, self.GREEN, screen_point, 3)


        if len(self.current_line) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_line]
            pygame.draw.lines(self.screen, self.YELLOW, False, screen_points, self.line_thickness)


        for polygon in self.polygons:
            if len(polygon) >= 3:
                screen_points = [self.get_screen_coords(p) for p in polygon]
                pygame.draw.polygon(self.screen, self.BLUE, screen_points)
                pygame.draw.polygon(self.screen, self.RED, screen_points, 2)
                
   
                if self.current_mode == "edit":
                    for point in polygon:
                        screen_point = self.get_screen_coords(point)
                        pygame.draw.circle(self.screen, self.WHITE, screen_point, 5)
                        pygame.draw.circle(self.screen, self.BLUE, screen_point, 3)


        if len(self.current_polygon) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_polygon]
            pygame.draw.lines(self.screen, self.YELLOW, False, screen_points, 2)


        for point in self.current_polygon:
            screen_point = self.get_screen_coords(point)
            pygame.draw.circle(self.screen, self.YELLOW, screen_point, 3)


        if self.current_mode == "edit" and self.selected_point:
            shape_type, shape_idx, point_idx = self.selected_point
            if shape_type == "line" and shape_idx < len(self.lines):
                point = self.lines[shape_idx][point_idx]
            elif shape_type == "polygon" and shape_idx < len(self.polygons):
                point = self.polygons[shape_idx][point_idx]
            else:
                point = None
                
            if point:
                screen_point = self.get_screen_coords(point)
                pygame.draw.circle(self.screen, self.YELLOW, screen_point, 8)
    
    def save_annotations(self):
        if not self.current_image or not self.image_files:
            return
        
        current_file = self.image_files[self.current_image_index]
        filename = current_file.stem
        
        orig_width = self.original_image.get_width()
        orig_height = self.original_image.get_height()
        

        lines_image = np.zeros((orig_height, orig_width), dtype=np.uint8)
        

        for line in self.lines:
            if len(line) >= 2:
                points = np.array(line, dtype=np.int32)
                cv2.polylines(lines_image, [points], False, 255, self.line_thickness)
        

        seg_image = np.zeros((orig_height, orig_width), dtype=np.uint8)
        

        for polygon in self.polygons:
            if len(polygon) >= 3:
                points = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(seg_image, [points], 255)
        

        try:
            cv2.imwrite(str(self.lines_folder / f"{filename}.png"), lines_image)
            cv2.imwrite(str(self.segmentation_folder / f"{filename}.png"), seg_image)
            print(f"Anotações guardada para {filename}")
        except Exception as e:
            print(f"Erro ao guardar anotações: {e}")
    
    def draw_ui(self):
        panel_rect = pygame.Rect(self.SCREEN_WIDTH - 180, 0, 180, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        
        y_pos = 20
        
        # Informações da imagem
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
        
        # Modo atual
        mode_text = f"Modo: {self.current_mode.upper()}"
        if self.current_mode == "line":
            mode_color = self.GREEN
        elif self.current_mode == "polygon":
            mode_color = self.BLUE
        else:  # edit
            mode_color = self.YELLOW
            
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
            "S - Guarda anotações",
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
            "EDIÇÃO:",
            "Click esq - Selecionar/mover",
            "Click dir - Deletar ponto",
            "Arrastar - Mover ponto",
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
                        modes = ["line", "polygon", "edit"]
                        current_idx = modes.index(self.current_mode)
                        self.current_mode = modes[(current_idx + 1) % len(modes)]
                        self.drawing = False
                        self.current_line = []
                        self.current_polygon = []
                        self.selected_point = None
                        self.dragging = False
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
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos, event.button)
                
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

