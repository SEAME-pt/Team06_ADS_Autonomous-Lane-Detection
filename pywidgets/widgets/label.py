
from pywidgets.core.widget import Widget

class Label(Widget):
    def __init__(self, x, y, width=0, height=0, text="", align=0, color=(255, 255, 255)):
        super().__init__(x, y, width, height)
        self.text = text
        self.align = align  # 'left', 'center', 'right'
        self.color = color

    def update(self, dt):
        text_width, text_height = self.skin.font.size(self.text)
        self.text_width = text_width
        self.text_height = text_height
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width = self.rect.width
        self.bound.height = self.rect.height

    def render(self, surface):
        if not self.visible:
            return

        font = self.skin.font
        x, y = self.get_x(), self.get_y()
        w, h = self.rect.width, self.rect.height

        if self.align == 1:
            text_x = x + (w - self.text_width) // 2
        elif self.align == 2:
            text_x = x + w - self.text_width - 4
        else:  # 'left'
            text_x = x + 4

        text_y = y + (h - self.text_height) // 2

        text_surface = font.render(self.text, True, self.color)
        surface.blit(text_surface, (text_x, text_y))

    def set_text(self, text):
        self.text = text