from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class Image(Widget):
    def __init__(self, x, y, image):
        self.image =  image.convert_alpha()
        self.resize = False
        self.color = (255,0,0)
        self.show_bound = True
        w, h = self.image.get_size()
        super().__init__(x, y, w, h)


    def set_size(self, width, height):
        self.image = pygame.transform.smoothscale(self.image, (width, height))
        self.rect.width  = width
        self.rect.height = height

    def on_resize(self, w, h):
        if self.resize:    
            self.image = pygame.transform.smoothscale(self.image, (w, h))
            self.rect.width  = w
            self.rect.height = h
    
    def set_image(self, image):
        self.image =  image
        w, h = self.image.get_size()
        self.rect.width  = w
        self.rect.height = h

    def render(self, surface):
        if self.visible:
            #if self.show_bound:
            pygame.draw.rect(self.image, self.color, self.bound)
            surface.blit(self.image, (self.get_x(), self.get_y()))

