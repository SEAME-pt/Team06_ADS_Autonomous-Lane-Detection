from pywidgets.core.widget import Widget
from pywidgets.core.skin import *
from pywidgets.widgets.button import Button

class ValueWidget(Widget):
    def __init__(self, x, y, width, height, value=0, min_value=0, max_value=100, 
                 step=1, on_change=None, format_str="{:.0f}"):
        super().__init__(x, y, width, height)
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.on_change = on_change
        self.format_str = format_str
        
        # Configuração de tamanhos
        button_width =30 #min(height, width // 5)

        
        # Criação dos componentes
        self.dec_button = Button(0, 0, button_width, height, "-", self._on_decrement)
        self.dec_button.parent = self
        self.inc_button = Button(width - button_width, 0, button_width, height, "+", self._on_increment)
        self.inc_button.parent = self
 
        
    def update(self, dt):
        super().update(dt)
        
        # Atualiza os subcomponentes
   
        
        self.dec_button.update(dt)
        self.inc_button.update(dt)
        
        #self.label.text = self._format_value()
 
        
    def render(self, surface):
        if not self.visible:
            return
        
        self.dec_button.skin = self.skin
        self.inc_button.skin = self.skin
        color = self.skin.get_property(SkinProps.BUTTON_HOVER_COLOR)
        rect = pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.rect.height)  
        pygame.draw.rect(surface, color, rect,  border_radius=8)

        tx = self.get_x() + (self.rect.width - self.skin.font.size(self._format_value())[0]) // 2
        ty = self.get_y() + (self.rect.height - self.skin.font.size(self._format_value())[1]) // 2
        text_surface = self.skin.font.render(self._format_value(), True, self.skin.get_property(SkinProps.TEXT_COLOR))
        surface.blit(text_surface, (tx, ty))
        
        self.dec_button.render(surface)
        self.inc_button.render(surface)
        
    
    def _format_value(self):
        return self.format_str.format(self.value)
    
    def _on_increment(self):
        new_value = min(self.value + self.step, self.max_value)
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)
    
    def _on_decrement(self):
        new_value = max(self.value - self.step, self.min_value)
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)
    
    def set_value(self, value):
        self.value = max(self.min_value, min(value, self.max_value))
        
    def get_value(self):
        return self.value
        
    
    def on_mouse_down(self, x, y):
        if not self.bound.collidepoint(x, y):
            return False
        if self.value < self.max_value:
            if self.inc_button.on_mouse_down(x, y):
                return True
        if self.value > self.min_value:
            if self.dec_button.on_mouse_down(x, y):
                return True
        return False
    
    def on_mouse_up(self, x, y):
        self.inc_button.on_mouse_up(x, y)
        self.dec_button.on_mouse_up(x, y)
        return False
    
    def on_mouse_move(self, x, y):
        if self.inc_button.on_mouse_move(x, y):
            return True

        if self.dec_button.on_mouse_move(x, y):
            return True
        return False