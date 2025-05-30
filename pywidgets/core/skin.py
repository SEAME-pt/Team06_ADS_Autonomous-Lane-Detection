
from enum import Enum   
import pygame  

class SkinProps:
 
    BACKGROUND_COLOR = 0
    DARK_BACKGROUND_COLOR = 1
    APP_COLOR = 2
 
    
 
    TEXT_COLOR = 10
    DISABLED_TEXT_COLOR = 11
    HIGHLIGHT_TEXT_COLOR = 12
   
    BUTTON_NORMAL_COLOR = 20
    BUTTON_HOVER_COLOR = 21
    BUTTON_PRESSED_COLOR = 22
    BUTTON_DISABLED_COLOR = 23
    
    TOGGLE_ON_COLOR = 90
    TOGGLE_OFF_COLOR = 91
    TOGGLE_SLIDER_COLOR = 92

 
    BORDER_COLOR = 30
    BORDER_HIGHLIGHT_COLOR = 31
    
 
    WINDOW_TITLE_BAR_COLOR = 40
    WINDOW_TITLE_TEXT_COLOR = 41
    WINDOW_RESIZE_COLOR = 42

    SPINNER_COLOR = 100
    KNOB_BACKGROUND = 101
    KNOB_INDICATOR = 102
    TEXTBOX_CURSOR_COLOR = 103
    LISTBOX_HIGHLIGHT_COLOR = 104
    LISTBOX_HOVER_COLOR = 105
    
 
    CHECKBOX_CHECK_COLOR = 50
    SLIDER_HANDLE_COLOR = 51
    PROGRESS_BAR_FILL_COLOR = 52
    SCROLLBAR_COLOR = 53
    RADIO_BUTTON_COLOR = 54
    RADIO_BUTTON_CHECK_COLOR = 55

    DROPDOWN_BACKGROUND = 93
    DROPDOWN_HOVER = 94
    DROPDOWN_BORDER = 95

     
    FONT_NAME = 60
    FONT_SIZE = 61
    TITLE_FONT_SIZE = 62
    
 
    BORDER_WIDTH = 70
    CORNER_RADIUS = 71
    WIDGET_PADDING = 72
    SHADOW_SIZE = 73
    
 
    SPACING = 80


class WidgetState:
    NORMAL = 0
    HOVER = 1
    PRESSED = 2
    DISABLED = 3
    ACTIVE = 4
    INACTIVE = 5
    FOCUSED = 6

class Margins:
    def __init__(self, left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __repr__(self):
        return f"Margins(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

class Skin:
    def __init__(self, name="default"):
        self.name = name
        self.properties = {}
        self.font = self.font = pygame.font.SysFont("Arial", 14)
        self.set_default_properties()
        
    def set_default_properties(self):
        self.properties.update({
            SkinProps.APP_COLOR: (30,30,30),
            SkinProps.BACKGROUND_COLOR: (200, 200, 200),
            SkinProps.DARK_BACKGROUND_COLOR: (220, 220, 220),
            SkinProps.TEXT_COLOR: (0, 0, 0),
            SkinProps.DISABLED_TEXT_COLOR: (150, 150, 150),
            SkinProps.HIGHLIGHT_TEXT_COLOR: (0, 120, 215),
            SkinProps.BUTTON_NORMAL_COLOR: (225, 225, 225),
            SkinProps.BUTTON_HOVER_COLOR: (235, 235, 235),
            SkinProps.BUTTON_PRESSED_COLOR: (200, 200, 200),
            SkinProps.BUTTON_DISABLED_COLOR: (210, 210, 210),
            SkinProps.BORDER_COLOR: (180, 180, 180),
            SkinProps.BORDER_HIGHLIGHT_COLOR: (0, 120, 215),
            SkinProps.WINDOW_TITLE_BAR_COLOR: (100, 100, 100),
            SkinProps.WINDOW_TITLE_TEXT_COLOR: (0, 0, 0),
            SkinProps.WINDOW_RESIZE_COLOR: (150, 150, 150),
            SkinProps.CHECKBOX_CHECK_COLOR: (0, 0, 0),
            SkinProps.SLIDER_HANDLE_COLOR: (80, 80, 80),
            SkinProps.PROGRESS_BAR_FILL_COLOR: (0, 120, 215),
            SkinProps.SCROLLBAR_COLOR: (180, 180, 180),
            SkinProps.RADIO_BUTTON_COLOR: (150, 150, 150),
            SkinProps.RADIO_BUTTON_CHECK_COLOR: (80, 80, 80),
            SkinProps.TOGGLE_ON_COLOR: (0, 160, 100),
            SkinProps.TOGGLE_OFF_COLOR: (160, 160, 160),
            SkinProps.TOGGLE_SLIDER_COLOR: (255, 255, 255),
            SkinProps.DROPDOWN_BACKGROUND: (230, 230, 230),
            SkinProps.DROPDOWN_HOVER: (210, 210, 210),
            SkinProps.DROPDOWN_BORDER: (180, 180, 180),
            SkinProps.SPINNER_COLOR: (0, 120, 215),
            SkinProps.KNOB_BACKGROUND: (100, 100, 100),
            SkinProps.KNOB_INDICATOR: (200, 200, 200),
            SkinProps.TEXTBOX_CURSOR_COLOR: (0, 0, 0),
            SkinProps.LISTBOX_HIGHLIGHT_COLOR: (100, 100, 255),
            SkinProps.FONT_NAME: "Arial",
            SkinProps.FONT_SIZE: 14,
            SkinProps.TITLE_FONT_SIZE: 18,
            SkinProps.BORDER_WIDTH: 1,
            SkinProps.CORNER_RADIUS: 8,
            SkinProps.WIDGET_PADDING: 5,
            SkinProps.SHADOW_SIZE: 0,
            SkinProps.SPACING: 5,
        })
    
    def set_property(self, property_id, value):
 
        self.properties[property_id] = value
        
    def set_properties(self, properties_dict):
 
        self.properties.update(properties_dict)
        
    def get_property(self, property_id, default=None):
   
        return self.properties.get(property_id, default)
    
    def copy(self):
 
        new_skin = Skin(f"{self.name}_copy")
        new_skin.properties = self.properties.copy()
        return new_skin
    
    def save(self, filename):
 
        import json
        
  
        string_properties = {str(k): v for k, v in self.properties.items()}
        
        with open(filename, 'w') as file:
            json.dump({
                "name": self.name,
                "properties": string_properties
            }, file, indent=4)
        
    @classmethod
    def load(cls, filename):
        """Carrega um skin a partir de um arquivo JSON."""
        import json
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                skin = cls(data.get("name", "loaded_skin"))
                
                # Converter chaves de string para números
                string_properties = data.get("properties", {})
                skin.properties = {int(k): v for k, v in string_properties.items()}
                
                return skin
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Erro ao carregar skin: {e}")
            return cls()  


class SkinManager:
    DEFAULT = 0
    DARK = 1
    COLORFUL = 2
    
    def __init__(self):
        self.skins = {}
        self.active_skin_id = self.DEFAULT
        self.create_default()
        
    def create_default(self):
        default_skin = Skin("default")
        self.add_skin(self.DEFAULT, default_skin)
        
        # Skin escuro
        dark_skin = Skin("dark")
        dark_skin.set_properties({
            SkinProps.BACKGROUND_COLOR: (30, 30, 30),
            SkinProps.DARK_BACKGROUND_COLOR: (40, 40, 40),
            SkinProps.TEXT_COLOR: (240, 240, 240),
            SkinProps.DISABLED_TEXT_COLOR: (150, 150, 150),
            SkinProps.HIGHLIGHT_TEXT_COLOR: (0, 160, 255),
            SkinProps.BUTTON_NORMAL_COLOR: (60, 60, 60),
            SkinProps.BUTTON_HOVER_COLOR: (70, 70, 70),
            SkinProps.BUTTON_PRESSED_COLOR: (50, 50, 50),
            SkinProps.BUTTON_DISABLED_COLOR: (45, 45, 45),
            SkinProps.BORDER_COLOR: (100, 100, 100),
            SkinProps.BORDER_HIGHLIGHT_COLOR: (0, 160, 255),
            SkinProps.WINDOW_TITLE_BAR_COLOR: (50, 50, 50),
            SkinProps.WINDOW_TITLE_TEXT_COLOR: (240, 240, 240),
            SkinProps.CHECKBOX_CHECK_COLOR: (240, 240, 240),
            SkinProps.SLIDER_HANDLE_COLOR: (100, 10, 100),
            SkinProps.PROGRESS_BAR_FILL_COLOR: (0, 160, 255),
            SkinProps.SCROLLBAR_COLOR: (80, 80, 80),
        })
        self.add_skin(self.DARK, dark_skin)
        
        # Skin colorido
        colorful_skin = Skin("colorful")
        colorful_skin.set_properties({
            SkinProps.BACKGROUND_COLOR: (230, 240, 255),
            SkinProps.DARK_BACKGROUND_COLOR: (210, 220, 240),
            SkinProps.TEXT_COLOR: (20, 20, 80),
            SkinProps.BUTTON_NORMAL_COLOR: (120, 180, 255),
            SkinProps.BUTTON_HOVER_COLOR: (140, 200, 255),
            SkinProps.BUTTON_PRESSED_COLOR: (100, 160, 240),
            SkinProps.BORDER_COLOR: (80, 140, 220),
            SkinProps.BORDER_HIGHLIGHT_COLOR: (255, 150, 50),
            SkinProps.WINDOW_TITLE_BAR_COLOR: (100, 150, 230),
            SkinProps.WINDOW_TITLE_TEXT_COLOR: (255, 255, 255),
            SkinProps.PROGRESS_BAR_FILL_COLOR: (255, 150, 50),
            SkinProps.CORNER_RADIUS: 8,
            SkinProps.BORDER_WIDTH: 2,
        })
        self.add_skin(self.COLORFUL, colorful_skin)
        
    def add_skin(self, skin_id, skin):
        self.skins[skin_id] = skin
        
    def get_skin(self, skin_id=None):
        if skin_id is None:
            skin_id = self.active_skin_id
        return self.skins.get(skin_id, self.skins.get(self.DEFAULT))
    
    def set_active_skin(self, skin_id):
        if skin_id in self.skins:
            self.active_skin_id = skin_id
            return True
        return False
    
    def load(self, filename, skin_id=None):
        skin = Skin.load(filename)
        if skin_id is None:
            skin_id = max(self.skins.keys(), default=-1) + 1
            
        self.add_skin(skin_id, skin)
        return skin_id
        
    def save(self, filename):
        active_skin = self.get_skin()
        active_skin.save(filename)