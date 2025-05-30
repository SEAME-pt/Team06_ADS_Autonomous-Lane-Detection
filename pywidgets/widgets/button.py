from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class Button(Widget):
    def __init__(self, x, y, width, height, text="", on_click=None):
        super().__init__(x, y, width, height)
        self.text = text
        self.on_click = on_click
        self.state = WidgetState.NORMAL
 



    def render(self, surface):
        if not self.visible:
            return

        x = self.get_x()
        y = self.get_y()
        w = self.bound.width
        h = self.bound.height
        rect = pygame.Rect(x, y, w, h)

        corner_radius = self.skin.get_property(SkinProps.CORNER_RADIUS) or 0
 

        if self.state == WidgetState.PRESSED:
            color = self.skin.get_property(SkinProps.BUTTON_PRESSED_COLOR)
        elif self.state == WidgetState.HOVER:
            color = self.skin.get_property(SkinProps.BUTTON_HOVER_COLOR)
        else:
            color = self.skin.get_property(SkinProps.BUTTON_NORMAL_COLOR)

        pygame.draw.rect(surface, color, rect,  border_radius=corner_radius)

        # Texto centralizado
        text_color = self.skin.get_property(SkinProps.TEXT_COLOR)
        text_surface = self.skin.font.render(self.text, True, text_color)
        text_width, text_height = self.skin.font.size(self.text)
        text_x = x + (w - text_width) // 2
        text_y = y + (h - text_height) // 2
        surface.blit(text_surface, (text_x, text_y))

    def on_mouse_move(self, x, y):
        if self.bound.collidepoint(x,y):
            self.state = WidgetState.HOVER
            return True
        else:
            self.state = WidgetState.NORMAL
            return False

    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x,y):
            self.state = WidgetState.PRESSED
            return True
        return False

    def on_mouse_up(self, x, y):
        if self.state == WidgetState.PRESSED:
            self.state = WidgetState.NORMAL
            if self.bound.collidepoint(x,y):
                if self.on_click:
                    self.on_click()
                return True
        return False