# PyWidgets — Modular UI Toolkit for Real-Time Autonomous Driving Systems

 **PyWidgets**, a fast, lightweight and extensible UI system built with **pygame** for real-time interaction with autonomous driving stacks like **CARLA**, **OpenCV**, and more.

> "Designed to help you visualize, debug, and control your autonomous pipeline in real time."

---

## 🚀 Features

- 🧩 **Modular Widgets**: Sliders, toggles, buttons, labels, checkboxes, dropdowns, lists, text inputs, and more.
- 📦 **Layouts & Containers**: HorizontalBox, VerticalBox, Foldouts, Grouping — auto-adjust and responsive.
- 🎨 **Themes & Skins**: Switch between dark/light/custom themes using ID-based skin properties.
- 🕹 **Live CARLA Integration**: Real-time image/mask stream widgets from CARLA sensors with adjustable transforms.
- 🧠 **Sensor-Friendly Design**: Includes camera visualization, semantic segmentation masks, and transform editing tools.
- 🔧 **Runtime Interaction**: Adjust values, tune parameters, and send control signals while your app runs.

---

## 📸 Showcase

| Layouts & Windows | Lists & Buttons | Foldouts & Controls | Text Editors |
|-------------------|------------------|----------------------|--------------|
| ![](images/box_layouts.gif) | ![](images/buttons.gif) | ![](images/foldout.gif) | ![](images/texts.gif) |
| Sliders & Toggles | Color Picker | Grouping UI | Dropdowns |
| ![](images/sliders.gif) | ![](images/color_picker.gif) | ![](images/groups.gif) | ![](images/lists.gif) |

---

## 💻 Example: Live CARLA Dashboard

```python
from pywidgets.core.app import App
from pywidgets.carla.Client import CarlaClient

class MyApp(App):
    def init(self):
        self.client = CarlaClient()
        image_widget = self.client.create_images(10, 50, 256, 256)
        mask_widget  = self.client.create_masks(270, 50, 256, 256)

        self.manager.add_widget(image_widget)
        self.manager.add_widget(mask_widget)
```

---

## 🛠 Getting Started

```bash
pip install pygame numpy
```

You will also need:
- [CARLA Simulator](https://carla.org/)
- Python 3.8+

---

## 📂 Project Structure

```plaintext
pyWidgets/
├── core/           # Base widget, manager, skin, events
├── widgets/        # All custom widgets (sliders, textviews, lists...)
├── layout/         # HorizontalBox, VerticalBox, Foldouts
├── carla/          # CARLA client integration + sensor surfaces
├── images/         # GIF demos of components
├── test_app.py     # Demo/test entry
├── README.md
```

---

## 📌 Roadmap

- [x] Fully custom GUI components
- [x] CARLA integration (images + semantic masks)
- [x] Live layout updates & responsive widgets
- [ ] Clipboard and file dialogs
- [ ] Remote control interfaces via socket/MQTT
- [ ] Auto-generate widgets from config/schema

---

## 🤝 Contributing

Feel free to open issues, suggest features, or fork & extend the toolkit. Contributions welcome!

## 📜 License

MIT License

---

Made with ❤️ for real-time autonomy development.
