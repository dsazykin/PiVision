import pyautogui
from pynput import keyboard, mouse

print("🔍 Input Identifier Started")
print("Press any key or mouse button to see its PyAutoGUI name.")
print("Press ESC to exit.\n")

def on_key_press(key):
    try:
        print(f"Keyboard input → '{key.char}'")
    except AttributeError:
        # Special keys like ctrl, alt, etc.
        print(f"Keyboard input → '{key}'")

    # Stop if ESC is pressed
    if key == keyboard.Key.esc:
        print("🛑 Exiting...")
        return False

def on_click(x, y, button, pressed):
    if pressed:
        if button == mouse.Button.left:
            name = "mouse_left"
        elif button == mouse.Button.right:
            name = "mouse_right"
        elif button == mouse.Button.middle:
            name = "mouse_middle"
        else:
            name = str(button)
        print(f"Mouse input → '{name}'")

# Set up listeners for both keyboard and mouse
with keyboard.Listener(on_press=on_key_press) as key_listener, \
     mouse.Listener(on_click=on_click) as mouse_listener:
    key_listener.join()
    mouse_listener.stop()
