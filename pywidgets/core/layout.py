import pygame
from .widget import Widget


class Layout(Widget):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.widgets = []
    
    def add(self, widget):
        widget.skin=self.skin
        widget.parent = self
        self.widgets.append(widget)
        return widget
    
    def update(self, dt):
        super().update(dt)
        for widget in self.widgets:
            if widget.enabled:
                widget.update(dt)
    
    def render(self, surface):
        for widget in self.widgets:
            if widget.visible:
                widget.render(surface)
    
    def debug(self, surface):
        for widget in self.widgets:
                widget.debug(surface)
    

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
        print("Layout on_resize")
        for widget in self.widgets:
            if widget.visible and widget.enabled:
                widget.on_resize(w,h)


