import pygame
from pywidgets.core.layout import Layout
from pywidgets.core.skin import *

class Window(Layout):
    def __init__(self, x, y, width, height,Title="Title"):
        super().__init__(x, y, width, height)
        self.dragging = False
        self.resizing = False
        self.drag_offset = (0, 0)
        self.resize_margin = 16
        self.minimize = False
        self.maximize = False
        self.closed = False
        self.top_bar_height = 30
        self.title = Title
        self.last_size = pygame.Rect(x, y, width, height)
        self.resize_zone = pygame.Rect(x, y, width, height)
        self.on_close = False
        self.on_minimize = False
        self.on_maximize = False
        self.on_resize_point = False
        self_on_resize= None
        self.radius = 13
        self._update_buttons()
        self.dirty = True
        

    def _update_buttons(self):
        x, y = self.get_x(), self.get_y()
        w = self.rect.width
        self.top_bar = pygame.Rect(x, y, w, self.top_bar_height)
        self.close_button = pygame.Rect(x + w - 25, y + 5, 20, 20)
        self.maximize_button = pygame.Rect(x + w - 50, y + 5, 20, 20)
        self.minimize_button = pygame.Rect(x + w - 75, y + 5, 20, 20)
        

    def update(self, dt):
        if not self.closed and not self.minimize:
            super().update(dt)

    def render(self, surface):
        if not self.visible or self.closed:
            return


        x = self.get_x()
        y = self.get_y()
        w = self.rect.width
        h = self.rect.height
        if self.dirty:
            self._update_buttons()
            self.on_resize(w,h)
            self.dirty = False
        
        rect = pygame.Rect(x,y, self.rect.width, self.rect.height)
        self._update_buttons()
        clip = surface.get_clip()
        if not self.minimize:
            surface.set_clip(self.bound)



        self.resize_zone = pygame.Rect(
            x + rect.width- self.resize_margin   ,
            y + rect.height-self.resize_margin   ,
            self.resize_margin,self.resize_margin)



              # Conteúdo
        if not self.minimize:
            pygame.draw.rect(surface, self.skin.get_property(SkinProps.BACKGROUND_COLOR), rect)
            super().render(surface)

        
        # Top bar
        pygame.draw.rect(surface, self.skin.get_property(SkinProps.WINDOW_TITLE_BAR_COLOR), self.top_bar)

        # Botões
        if self.on_close:
            pygame.draw.circle(surface, (255, 0, 0), self.close_button.center, self.radius)
        
        pygame.draw.rect(surface, (220, 0, 0), self.close_button)     # Fechar

        if self.on_maximize:
            pygame.draw.circle(surface, (0, 255, 0), self.maximize_button.center, self.radius)

        pygame.draw.rect(surface, (0, 200, 0), self.maximize_button)  # Maximize
        if self.on_minimize:
            pygame.draw.circle(surface, (255, 255, 0), self.minimize_button.center, self.radius)
        pygame.draw.rect(surface, (200, 200, 0), self.minimize_button)  # Minimizar

        text_surface = self.skin.font.render(self.title, True,self.skin.get_property(SkinProps.WINDOW_TITLE_TEXT_COLOR))
        _, text_height = self.skin.font.size(self.title)
        text_x =10 + x 
        text_y =(self.top_bar_height - text_height) // 2 +y
        surface.blit(text_surface, (text_x, text_y))

      


        if not self.minimize:
            surface.set_clip(clip)
        if self.on_resize_point:
            resize_triangle = [
            (self.bound.right, self.bound.bottom),                       # canto
            (self.bound.right - self.resize_margin, self.bound.bottom),                 # esquerda
            (self.bound.right, self.bound.bottom - self.resize_margin)                  # cima
        ]
            pygame.draw.polygon(surface, self.skin.get_property(SkinProps.WINDOW_RESIZE_COLOR), resize_triangle)

 
 


    def on_mouse_down(self, x, y):
        if self.closed:
            return False

        if self.close_button.collidepoint(x, y):
            self.closed = True
            return True

        if self.minimize_button.collidepoint(x, y):
            self.minimize = not self.minimize
            return True

        if self.maximize_button.collidepoint(x, y):
            self.maximize = not self.maximize
            if self.maximize:
                width, height = pygame.display.get_window_size()
                self.rect = pygame.Rect(0, 0, width, height)
                if (self.minimize):
                    self.minimize = False
            else: 
                self.rect = self.last_size
            return True

        if self.top_bar.collidepoint(x, y) and  not self.maximize:
            self.dragging = True
            self.drag_offset = (x - self.get_x(), y - self.get_y())
            return True

       

        if  self.on_resize_point and not self.minimize and not self.maximize:
            self.resizing = True
            self.drag_offset = (x - self.get_x(), y - ( self.get_y()+self.top_bar_height))
            return True

        return super().on_mouse_down(x, y)

    def on_mouse_up(self, x, y):
        self.dragging = False
        self.resizing = False
        return super().on_mouse_up(x, y)

    def on_mouse_move(self, x, y):
        self.on_minimize=self.minimize_button.collidepoint(x, y)
        self.on_maximize=self.maximize_button.collidepoint(x, y)
        self.on_close= self.close_button.collidepoint(x, y)
        self.on_resize_point= self.resize_zone.collidepoint(x, y) 

        if self.dragging:
            new_x = x - self.drag_offset[0]
            new_y = y - self.drag_offset[1]
            self.set_position(new_x, new_y)
            return True

        if self.resizing:
            new_width = max(100, x - self.get_x())
            new_height = max(60, y - self.get_y())
            self.rect.width = new_width
            self.rect.height = new_height
            self.last_size = self.bound
            self.on_resize(new_width, new_height)
            return True

        return super().on_mouse_move(x, y)

    def on_resize(self, width, height):
        for w in self.widgets:
            w.on_resize(width, height)

    def update(self, dt):
        if not self.closed and not self.minimize:
            super().update(dt)
