import pygame
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk

# Importar pyWidgets baseado nos exemplos fornecidos
from pywidgets.core.app import App
from pywidgets.layout.window import Window
from pywidgets.widgets.button import Button
from pywidgets.widgets.slider import Slider
from pywidgets.widgets.check_box import CheckGroup
from pywidgets.widgets.radio import RadioGroup
from pywidgets.widgets.toggle import ToggleSwitch
from pywidgets.widgets.value import ValueWidget
from pywidgets.widgets.progress_bar import ProgressBar
from pywidgets.widgets.foldout import FoldoutGroup, Foldout

class AnnotationToolApp(App):
    def init(self):
        # Configurações da aplicação (mantendo as originais)
        self.SCREEN_WIDTH = 1400
        self.SCREEN_HEIGHT = 900
        
        # Cores (mantendo as originais)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Estado da aplicação (mantendo a lógica original)
        self.current_mode = "line"  # "line", "polygon", "edit"
        self.images_folder = None
        self.image_files = []
        self.current_image_index = 0
        self.current_image = None
        self.original_image = None
        self.image_scale = 1.0
        self.image_offset = [0, 0]
        
        # Desenho (mantendo a lógica original)
        self.drawing = False
        self.current_line = []
        self.lines = []
        self.current_polygon = []
        self.polygons = []
        self.line_thickness = 3
        
        # Edição (funcionalidades que você já tinha)
        self.selected_point = None
        self.dragging = False
        self.selection_radius = 10
        
        # Configurações adicionais
        self.show_points = True
        self.auto_save = False
        
        # Criar interface
        self.setup_ui()
        self.setup_output_folders()
    
    def setup_ui(self):
        """Configura a interface usando pyWidgets"""
        
        # === JANELA PRINCIPAL DE CONTROLES ===
        self.control_window = self.manager.add_widget(
            Window(self.SCREEN_WIDTH - 320, 20, 300, 600, Title="🎨 Controles")
        )
        
        # Botão para selecionar pasta
        btn_folder = self.control_window.add(
            Button(20, 40, 250, 35, "📁 Selecionar Pasta")
        )
        btn_folder.on_click = self.select_folder
        
        # Navegação de imagens
        btn_prev = self.control_window.add(
            Button(20, 85, 120, 30, "◀ Anterior")
        )
        btn_prev.on_click = self.prev_image
        
        btn_next = self.control_window.add(
            Button(150, 85, 120, 30, "Próxima ▶")
        )
        btn_next.on_click = self.next_image
        
        # RadioGroup para modo de desenho
        self.radio_mode = self.control_window.add(
            RadioGroup(20, 130, 250, 30, ["🖊️ Linha", "🔷 Polígono", "✏️ Edição"])
        )
        self.radio_mode.on_change = self.change_mode
        
        # Slider para espessura da linha
        self.slider_thickness = self.control_window.add(
            Slider(20, 240, 200, vertical=False, 
                   min_value=1, max_value=10, value=3,
                   on_change=self.change_thickness)
        )
        
        # Slider para raio de seleção
        self.slider_selection = self.control_window.add(
            Slider(20, 280, 200, vertical=False,
                   min_value=5, max_value=25, value=10,
                   on_change=self.change_selection_radius)
        )
        
        # CheckGroup para opções
        self.check_options = self.control_window.add(
            CheckGroup(20, 320, 250, 30, ["Mostrar Pontos", "Auto Salvar"])
        )
        self.check_options.on_change = self.toggle_options
        
        # Botões de ação
        btn_save = self.control_window.add(
            Button(20, 420, 250, 35, "💾 Salvar Anotações")
        )
        btn_save.on_click = self.save_annotations
        
        btn_clear = self.control_window.add(
            Button(20, 465, 250, 35, "🗑️ Limpar Tudo")
        )
        btn_clear.on_click = self.clear_annotations
        
        btn_export = self.control_window.add(
            Button(20, 510, 250, 35, "📤 Exportar Dataset")
        )
        btn_export.on_click = self.export_dataset
        
        # === JANELA DE ESTATÍSTICAS ===
        self.stats_window = self.manager.add_widget(
            Window(20, 20, 300, 250, Title="📊 Estatísticas")
        )
        
        # Progress bar para progresso do dataset
        self.progress_dataset = self.stats_window.add(
            ProgressBar(20, 50, 250, 25, vertical=False, 
                       value=0.0, show_text=True, mode=0)
        )
        
        # ValueWidgets para contadores
        self.value_lines = self.stats_window.add(
            ValueWidget(20, 100, 80, 25, value=0, min_value=0, max_value=999)
        )
        
        self.value_polygons = self.stats_window.add(
            ValueWidget(120, 100, 80, 25, value=0, min_value=0, max_value=999)
        )
        
        self.value_points = self.stats_window.add(
            ValueWidget(220, 100, 80, 25, value=0, min_value=0, max_value=9999)
        )
        
        # === JANELA DE INFORMAÇÕES ===
        self.info_window = self.manager.add_widget(
            Window(20, 290, 300, 200, Title="ℹ️ Informações")
        )
    
    # === CALLBACKS DOS WIDGETS ===
    
    def select_folder(self, button=None):
        """Callback para selecionar pasta (mantendo a lógica original)"""
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if folder:
            self.images_folder = Path(folder)
            self.load_images()
        root.destroy()
    
    def change_mode(self, radio_group, selected_index):
        """Callback para mudança de modo"""
        modes = ["line", "polygon", "edit"]
        self.current_mode = modes[selected_index]
        self.drawing = False
        self.current_line = []
        self.current_polygon = []
        self.selected_point = None
        self.dragging = False
    
    def change_thickness(self, value):
        """Callback para mudança de espessura"""
        self.line_thickness = int(value)
    
    def change_selection_radius(self, value):
        """Callback para mudança do raio de seleção"""
        self.selection_radius = int(value)
    
    def toggle_options(self, check_group, checked_items):
        """Callback para opções"""
        self.show_points = "Mostrar Pontos" in checked_items
        self.auto_save = "Auto Salvar" in checked_items
    
    def prev_image(self, button=None):
        """Callback para imagem anterior (mantendo a lógica original)"""
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def next_image(self, button=None):
        """Callback para próxima imagem (mantendo a lógica original)"""
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
    
    def clear_annotations(self, button=None):
        """Callback para limpar anotações (mantendo a lógica original)"""
        self.lines = []
        self.polygons = []
        self.current_line = []
        self.current_polygon = []
        self.drawing = False
        self.selected_point = None
        self.dragging = False
        self.update_statistics()
    
    def export_dataset(self, button=None):
        """Callback para exportar dataset completo"""
        if not self.image_files:
            return
        
        total_images = len(self.image_files)
        for i, image_file in enumerate(self.image_files):
            self.current_image_index = i
            self.load_current_image()
            self.save_annotations()
            
            # Atualiza progress bar
            progress = (i + 1) / total_images
            self.progress_dataset.set_value(progress)
        
        print(f"Dataset exportado: {total_images} imagens processadas")
    
    # === MÉTODOS ORIGINAIS (mantendo toda a lógica) ===
    
    def setup_output_folders(self):
        """Cria as pastas de output se não existirem"""
        self.lines_folder = Path("output/lines")
        self.segmentation_folder = Path("output/segmentation")
        self.lines_folder.mkdir(parents=True, exist_ok=True)
        self.segmentation_folder.mkdir(parents=True, exist_ok=True)
    
    def load_images(self):
        """Carrega todas as imagens da pasta selecionada"""
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
        """Carrega a imagem atual"""
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
        """Ajusta a imagem para caber na tela"""
        if not self.original_image:
            return
        
        available_width = self.SCREEN_WIDTH - 340
        available_height = self.SCREEN_HEIGHT - 100
        
        img_width = self.original_image.get_width()
        img_height = self.original_image.get_height()
        
        scale_x = available_width / img_width
        scale_y = available_height / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)
        
        new_width = int(img_width * self.image_scale)
        new_height = int(img_height * self.image_scale)
        self.current_image = pygame.transform.scale(self.original_image, (new_width, new_height))
        
        self.image_offset = [
            (available_width - new_width) // 2,
            (available_height - new_height) // 2 + 50
        ]
    
    def update_statistics(self):
        """Atualiza estatísticas na interface"""
        total_points = sum(len(line) for line in self.lines) + sum(len(poly) for poly in self.polygons)
        
        self.value_lines.value = len(self.lines)
        self.value_polygons.value = len(self.polygons)
        self.value_points.value = total_points
        
        if self.image_files:
            progress = (self.current_image_index + 1) / len(self.image_files)
            self.progress_dataset.set_value(progress)
    
    # === MÉTODOS DE COORDENADAS (mantendo a lógica original) ===
    
    def get_image_coords(self, screen_pos):
        """Converte coordenadas da tela para coordenadas da imagem original"""
        if not self.current_image:
            return None
        
        x, y = screen_pos
        img_x = x - self.image_offset[0]
        img_y = y - self.image_offset[1]
        
        if (img_x < 0 or img_y < 0 or 
            img_x >= self.current_image.get_width() or 
            img_y >= self.current_image.get_height()):
            return None
        
        orig_x = int(img_x / self.image_scale)
        orig_y = int(img_y / self.image_scale)
        
        return (orig_x, orig_y)
    
    def get_screen_coords(self, image_pos):
        """Converte coordenadas da imagem original para coordenadas da tela"""
        orig_x, orig_y = image_pos
        img_x = int(orig_x * self.image_scale)
        img_y = int(orig_y * self.image_scale)
        screen_x = img_x + self.image_offset[0]
        screen_y = img_y + self.image_offset[1]
        return (screen_x, screen_y)
    
    def find_nearest_point(self, pos):
        """Encontra o ponto mais próximo do cursor"""
        image_pos = self.get_image_coords(pos)
        if not image_pos:
            return None
        
        min_distance = float('inf')
        nearest_point = None
        
        # Verifica pontos das linhas
        for line_idx, line in enumerate(self.lines):
            for point_idx, point in enumerate(line):
                screen_point = self.get_screen_coords(point)
                distance = ((screen_point[0] - pos[0])**2 + (screen_point[1] - pos[1])**2)**0.5
                if distance < self.selection_radius and distance < min_distance:
                    min_distance = distance
                    nearest_point = ("line", line_idx, point_idx)
        
        # Verifica pontos dos polígonos
        for poly_idx, polygon in enumerate(self.polygons):
            for point_idx, point in enumerate(polygon):
                screen_point = self.get_screen_coords(point)
                distance = ((screen_point[0] - pos[0])**2 + (screen_point[1] - pos[1])**2)**0.5
                if distance < self.selection_radius and distance < min_distance:
                    min_distance = distance
                    nearest_point = ("polygon", poly_idx, point_idx)
        
        return nearest_point
    
    def delete_point(self, point_info):
        """Deleta um ponto específico"""
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
    
    # === MÉTODOS DE DESENHO (mantendo a lógica original) ===
    
    def draw_annotations(self, surface):
        """Desenha todas as anotações na tela"""
        if not self.current_image:
            return
        
        # Desenha linhas finalizadas
        for line in self.lines:
            if len(line) >= 2:
                screen_points = [self.get_screen_coords(p) for p in line]
                pygame.draw.lines(surface, self.GREEN, False, screen_points, self.line_thickness)
                
                # Desenha pontos editáveis no modo de edição
                if self.current_mode == "edit" and self.show_points:
                    for point in line:
                        screen_point = self.get_screen_coords(point)
                        pygame.draw.circle(surface, self.WHITE, screen_point, 5)
                        pygame.draw.circle(surface, self.GREEN, screen_point, 3)
        
        # Desenha linha atual sendo desenhada
        if len(self.current_line) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_line]
            pygame.draw.lines(surface, self.YELLOW, False, screen_points, self.line_thickness)
        
        # Desenha polígonos finalizados
        for polygon in self.polygons:
            if len(polygon) >= 3:
                screen_points = [self.get_screen_coords(p) for p in polygon]
                pygame.draw.polygon(surface, self.BLUE, screen_points)
                pygame.draw.polygon(surface, self.RED, screen_points, 2)
                
                # Desenha pontos editáveis no modo de edição
                if self.current_mode == "edit" and self.show_points:
                    for point in polygon:
                        screen_point = self.get_screen_coords(point)
                        pygame.draw.circle(surface, self.WHITE, screen_point, 5)
                        pygame.draw.circle(surface, self.BLUE, screen_point, 3)
        
        # Desenha polígono atual sendo desenhado
        if len(self.current_polygon) >= 2:
            screen_points = [self.get_screen_coords(p) for p in self.current_polygon]
            pygame.draw.lines(surface, self.YELLOW, False, screen_points, 2)
        
        # Desenha pontos do polígono atual
        for point in self.current_polygon:
            screen_point = self.get_screen_coords(point)
            pygame.draw.circle(surface, self.YELLOW, screen_point, 3)
        
        # Destaca ponto selecionado
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
                pygame.draw.circle(surface, self.YELLOW, screen_point, 8)
    
    def save_annotations(self, button=None):
        """Salva as anotações como imagens (mantendo a lógica original)"""
        if not self.current_image or not self.image_files:
            return
        
        current_file = self.image_files[self.current_image_index]
        filename = current_file.stem
        
        # Dimensões da imagem original
        orig_width = self.original_image.get_width()
        orig_height = self.original_image.get_height()
        
        # Cria imagem para linhas (fundo preto, linhas brancas)
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
        
        # Salva as imagens
        try:
            cv2.imwrite(str(self.lines_folder / f"{filename}.png"), lines_image)
            cv2.imwrite(str(self.segmentation_folder / f"{filename}.png"), seg_image)
            print(f"Anotações salvas para {filename}")
            self.update_statistics()
        except Exception as e:
            print(f"Erro ao salvar anotações: {e}")
    
    # === MÉTODOS DE EVENTOS (mantendo a lógica original) ===
    
    def handle_mouse_click(self, pos, button):
        """Manipula cliques do mouse (mantendo a lógica original)"""
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
            elif button == 3:  # Botão direito - deletar ponto
                nearest_point = self.find_nearest_point(pos)
                if nearest_point:
                    self.delete_point(nearest_point)
                    self.update_statistics()
        
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
                    self.update_statistics()

        elif self.current_mode == "polygon":
            if button == 1:  # Botão esquerdo
                self.current_polygon.append(image_pos)
            elif button == 3:  # Botão direito - finaliza polígono
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                    self.current_polygon = []
                    self.update_statistics()
    
    def handle_mouse_up(self, pos, button):
        """Manipula quando o mouse é solto"""
        if button == 1 and self.dragging:
            self.dragging = False
            self.selected_point = None
    
    def handle_mouse_motion(self, pos):
        """Manipula movimento do mouse (mantendo a lógica original)"""
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
    
    # === MÉTODOS DO PYWIDGETS ===
    
    def on_render(self, surface):
        """Método de renderização do pyWidgets"""
        # Limpa a tela
        #surface.fill(self.BLACK)
        
        # Desenha a imagem
        if self.current_image:
            surface.blit(self.current_image, self.image_offset)
        
        # Desenha anotações
        self.draw_annotations(surface)
        
        # Auto salvar se habilitado
        if self.auto_save and (len(self.lines) > 0 or len(self.polygons) > 0):
            self.save_annotations()
    
    def on_event(self, event):
        """Manipula eventos personalizados"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Só processar cliques na área da imagem (fora das janelas de widgets)
            if event.pos[0] < self.SCREEN_WIDTH - 340:
                self.handle_mouse_click(event.pos, event.button)
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.pos[0] < self.SCREEN_WIDTH - 340:
                self.handle_mouse_up(event.pos, event.button)
        
        elif event.type == pygame.MOUSEMOTION:
            if event.pos[0] < self.SCREEN_WIDTH - 340:
                self.handle_mouse_motion(event.pos)
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and pygame.key.get_pressed()[pygame.K_LCTRL]:
                self.save_annotations()

if __name__ == "__main__":
    app = AnnotationToolApp()
    app.run()

