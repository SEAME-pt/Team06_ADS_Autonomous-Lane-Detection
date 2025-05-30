import pygame

class Widget:
    def __init__(self, x, y, width, height, parent=None, skin=None):
        self.rect  = pygame.Rect(x, y, width, height)
        self.bound = pygame.Rect(x, y, width, height)
        self.visible = True
        self.enabled = True
        self.skin = skin
        self.parent = parent
    

    def get_x(self):
        if self.parent is not None:
            return self.parent.get_x() + self.rect.x
        return self.rect.x
    
    def get_y(self):
        if self.parent is not None:
            return self.parent.get_y() + self.rect.y
        return self.rect.y
    

  

    
    def debug(self, surface):
         pygame.draw.rect(surface, (255, 0, 0), self.bound, 1) 

    def update(self, dt):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width  = self.rect.width
        self.bound.height = self.rect.height  

    def render(self, surface):
        pass

   
    
    def set_position(self, x, y):
        self.rect.x = x
        self.rect.y = y
    
    def set_size(self, width, height):
        self.rect.width = width
        self.rect.height = height
    
    def set_visible(self, visible):
        self.visible = visible

    def on_mouse_down(self,x,y):
        pass

    def on_mouse_up(self,x,y):
        pass

    def on_mouse_move(self,x,y):
        pass

    def on_key_down(self,key):
        pass

    def on_key_up(self,key):
        pass
    

    def on_mouse_wheel(self,delta):
        pass

    def on_resize(self,w,h):
        pass


