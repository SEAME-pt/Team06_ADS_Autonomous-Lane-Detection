
from pywidgets.core.widget import Widget
from pywidgets.core.skin import *

class RadioItem(Widget):
    def __init__(self, x, y, radius=12, text="", checked=False,  on_change=None):
        super().__init__(x, y, radius*2, radius*2)
        self.radius = radius
        self.text = text
        self.checked = checked 
        self.on_change = on_change

    

    def render(self, surface):
        if not self.visible:
            return
        cx = self.get_x() + self.radius
        cy = self.get_y() + self.radius
        text_width, text_height = self.skin.font.size(self.text) 
        self.rect.width  = text_width + self.radius * 2 + 8
        self.rect.height = text_height+ self.radius * 2
        self.bound = pygame.Rect(self.get_x(), self.get_y(), self.rect.width, self.rect.height)  
        #  externo
        border_color = self.skin.get_property(SkinProps.RADIO_BUTTON_COLOR,(0, 0, 0))
        pygame.draw.circle(surface, border_color, (cx, cy), self.radius, 2)
        #  interno se selecionado
        if self.checked:
            fill_color = self.skin.get_property(SkinProps.RADIO_BUTTON_CHECK_COLOR,(0, 0, 0))
            pygame.draw.circle(surface, fill_color, (cx, cy), self.radius - 4)

        if self.text:
            text_surface = self.skin.font.render(self.text, True,self.skin.get_property(SkinProps.TEXT_COLOR))
            surface.blit(text_surface, (cx + self.radius + 8, cy - text_surface.get_height() // 2))

        #pygame.draw.rect(surface, (255,0,255), self.bound, width=1)
        #pygame.draw.rect(surface, (255,0,0), self.rect, width=1)

    def on_mouse_down(self, x, y):
        return self.bound.collidepoint(x, y)
        

    def set_checked(self, value):
        self.checked = value
        if self.on_change:
            self.on_change(self.checked)
    
    






class RadioGroup(Widget):
    def __init__(self, x, y, width, height, items,  max_columns=2, spacing=8, selected=0, on_change=None):
        super().__init__(x, y, width, height)
        self.items = []
        self.selected = selected
        self.max_columns = max_columns
        self.spacing = spacing
        self.on_change = on_change
        self._create_items(items)
        self._align_items()

    def _create_items(self, labels):
        font = self.skin.font if self.skin else pygame.font.SysFont("Arial", 14)
        self.max_widths = []
        self.max_height = 0

        for idx, label in enumerate(labels):
            text_width, text_height = font.size(label)
            total_width = 24 + 8 + text_width  # 24px para círculo, 8px de espaço
            self.max_widths.append(total_width)
            self.max_height = max(self.max_height, text_height, 24)

            radio_item =  RadioItem(0, 0, text=label, checked=(idx==self.selected))
            radio_item.on_change = self.on_change
            radio_item.group = self
            radio_item.skin = self.skin
            radio_item.parent = self
            self.items.append(radio_item)
        
 
        self.widget_width = max(self.max_widths)
        self.widget_height = self.max_height + 8  # padding vertical

    def _align_items(self):
 
        for idx, item in enumerate(self.items):
            row = idx // self.max_columns
            col = idx % self.max_columns
            x = col * (self.widget_width + self.spacing)
            y = row * (self.widget_height + self.spacing)
            item.set_position(x, y)

 
        total_cols = min(self.max_columns, len(self.items))
        total_rows = (len(self.items) + self.max_columns - 1) // self.max_columns
        self.set_size(
            total_cols * self.widget_width + (total_cols - 1) * self.spacing,
            total_rows * self.widget_height + (total_rows - 1) * self.spacing
        )

    def select(self, selected_item):
        for idx, item in enumerate(self.items):
            item.checked = (item == selected_item)
            if item == selected_item:
                self.selected = idx
        if self.on_change:
            self.on_change(self.selected)

 

    def render(self, surface):
        if not self.visible:
            return
        #pygame.draw.rect(surface, (255,0,0), self.bound, width=1)


    
            
        for item in self.items:
            item.skin = self.skin
            item.render(surface)

    def on_mouse_move(self, x, y):
 
        
        return super().on_mouse_move(x, y)  

    def on_mouse_down(self, x, y):
        if not self.bound.collidepoint(x, y):
           return False
     
        for idx, item in enumerate(self.items):
            if item.on_mouse_down(x,y):
                self.select(self.items[idx])
                if self.on_change:
                    self.on_change(self.selected)
                return True


        return False

 