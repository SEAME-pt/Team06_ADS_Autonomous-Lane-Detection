import pygame
from .widget import *
from .skin import SkinManager
class WidgetManager:
    def __init__(self, surface):
        self.surface = surface
        self.widgets = []
        self.skin_manager = SkinManager()
        self.bound = self.surface.get_size()
        self.width=0
        self.height = 0
        
        
    def set_skin(self, skin_id):
        if self.skin_manager.set_active_skin(skin_id):
            self._update_widgets_skin()
            return True
        return False
    
    def _update_widgets_skin(self):
        active_skin = self.skin_manager.get_skin()
        for widget in self.widgets:
            widget.update_skin(active_skin)
    
    def add_widget(self, widget):
        widget.skin=self.skin_manager.get_skin()
        self.widgets.append(widget)
        return widget
    
    def update(self, dt):
        
        self.width, self.height = pygame.display.get_window_size()
        self.bound = pygame.Rect(0, 0, self.width, self.height)
        #self.surface.set_clip(self.bound)
        for widget in self.widgets:
            if widget.enabled:
                widget.update(dt)
    
    def render(self):
        for widget in self.widgets:
            if widget.visible:
                widget.render(self.surface)
    
    def debug(self):
        for widget in self.widgets:
                widget.debug(self.surface)
    
    def load_skin(self, filename):
        skin_id = self.skin_manager.load(filename)
        return self.set_skin(skin_id)
    

    def on_mouse_down(self, x, y):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_mouse_down(x, y):
                    break


    def on_mouse_up(self,x,y):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_mouse_up(x, y):
                    break

    def on_mouse_move(self,x,y):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_mouse_move(x, y):
                    break

    def on_key_down(self,key):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_key_down(key):
                    break

    def on_key_up(self,key):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_key_up(key):
                    break
    

    def on_mouse_wheel(self,delta):
        for widget in reversed(self.widgets):  
            if widget.visible and widget.enabled:
                if widget.on_mouse_wheel(delta):
                    break

    def on_resize(self,w,h):
        for widget in self.widgets:
            widget.on_resize(w,h)

