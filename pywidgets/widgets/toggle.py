from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class ToggleSwitch(Widget):
    def __init__(self, x, y, width=80, height=24, state=False, on_change=None, text_on="ON", text_off="OFF", show_text=True):
        super().__init__(x, y, width, height)
        self.state = state
        self.on_change = on_change
        self.text_on = text_on
        self.text_off = text_off
        self.show_text = show_text


    def render(self, surface):
        if not self.visible:
            return
        
        x, y = self.get_x(), self.get_y()
        w, h = self.rect.width, self.rect.height
        radius = h // 2

        # Cores do skin
        on_color = self.skin.get_property(SkinProps.TOGGLE_ON_COLOR)
        off_color = self.skin.get_property(SkinProps.TOGGLE_OFF_COLOR)
        slider_color = self.skin.get_property(SkinProps.TOGGLE_SLIDER_COLOR)
        
        track_color = on_color if self.state else off_color

        # Fundo (trilho)
        pygame.draw.rect(surface, track_color, (x, y, w, h), border_radius=radius)

        # Posição do círculo
        pad = 3
        cx = x + (w - h + pad) if self.state else x + pad
        cy = y + pad
        pygame.draw.circle(surface, slider_color, (cx + (h - 2 * pad)//2, cy + (h - 2 * pad)//2), (h - 2 * pad)//2)
        if self.show_text:
            text = self.text_on if self.state else self.text_off
            font = self.skin.font
            text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
            text_surface = font.render(text, True, text_color)
            tx = x + (w - text_surface.get_width()) // 2
            ty = y + (h - text_surface.get_height()) // 2
            surface.blit(text_surface, (tx, ty))


    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x, y):
            self.state = not self.state
            if self.on_change:
                self.on_change(self.state)
            return True
        return False

    def on_mouse_move(self, x, y):
        self.hover = self.bound.collidepoint(x, y)
        return self.hover

    def set_state(self, state):
        if self.state != state:
            self.state = state
            if self.on_change:
                self.on_change(self.state)

    def get_state(self):
        return self.state