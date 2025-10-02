import tkinter as tk
from tkinter import messagebox

# ====== Default gesture mappings ======
gesture_mappings = {
    "Swipe Left": "a",
    "Swipe Right": "d",
    "Fist": "Enter",
    "Open Palm": "Space",
}

# ====== Login Screen ======
def show_login_screen():
    login_root = tk.Tk()
    login_root.title("Login")
    login_root.geometry("250x150")

    tk.Label(login_root, text="Username:").pack(pady=(10, 0))
    entry_username = tk.Entry(login_root)
    entry_username.pack()

    tk.Label(login_root, text="Password:").pack(pady=(10, 0))
    entry_password = tk.Entry(login_root, show="*")
    entry_password.pack()

    def login():
        username = entry_username.get()
        password = entry_password.get()
        if username == "admin" and password == "1234":
            login_root.destroy()
            show_welcome_screen(username)
        else:
            messagebox.showerror("Login Failed", "Invalid credentials!")

    tk.Button(login_root, text="Login", command=login).pack(pady=10)
    login_root.mainloop()

# ====== Welcome Screen ======
def show_welcome_screen(username):
    welcome_root = tk.Tk()
    welcome_root.title("Welcome")
    welcome_root.geometry("350x250")

    tk.Label(welcome_root, text=f"Welcome, {username}!", font=("Arial", 16)).pack(pady=15)

    keystroke_label = tk.Label(welcome_root, text="Press any key...", font=("Arial", 12))
    keystroke_label.pack()

    def on_key_press(event):
        keystroke_label.config(text=f"You pressed: {repr(event.char)} (KeyCode: {event.keycode})")

    welcome_root.bind("<Key>", on_key_press)

    def logout():
        welcome_root.destroy()
        show_login_screen()

    def open_mapping_screen():
        welcome_root.destroy()
        show_mapping_screen(username)

    tk.Button(welcome_root, text="Change Mappings", command=open_mapping_screen).pack(pady=10)
    tk.Button(welcome_root, text="Log Out", command=logout).pack(pady=5)

    welcome_root.mainloop()

# ====== Mapping Screen ======
def show_mapping_screen(username):
    map_root = tk.Tk()
    map_root.title("Gesture Mapping")
    map_root.geometry("400x350")

    tk.Label(map_root, text="Gesture to Key Mappings", font=("Arial", 14)).pack(pady=10)

    mapping_frames = {}

    # Temporary state to track which gesture is waiting for input
    waiting_for_input = {"gesture": None}

    # Callback to bind key to gesture
    def on_key_press(event):
        gesture = waiting_for_input["gesture"]
        if gesture:
            new_key = event.keysym
            gesture_mappings[gesture] = new_key
            label = mapping_frames[gesture]["label"]
            label.config(text=f"Key: {new_key}")
            waiting_for_input["gesture"] = None
            map_root.unbind("<Key>")
            messagebox.showinfo("Updated", f"'{gesture}' is now mapped to '{new_key}'.")

    # Generate UI rows
    for gesture, key in gesture_mappings.items():
        frame = tk.Frame(map_root)
        frame.pack(pady=5)

        tk.Label(frame, text=gesture + ":", width=15, anchor="w").pack(side="left")

        key_label = tk.Label(frame, text=f"Key: {key}", width=10)
        key_label.pack(side="left", padx=5)

        def make_change_func(gesture_name):
            def change():
                waiting_for_input["gesture"] = gesture_name
                messagebox.showinfo("Key Bind", f"Press a key to bind to '{gesture_name}'")
                map_root.bind("<Key>", on_key_press)
            return change

        change_button = tk.Button(frame, text="Change", command=make_change_func(gesture))
        change_button.pack(side="left")

        mapping_frames[gesture] = {"label": key_label, "button": change_button}

    def save_and_return():
        map_root.destroy()
        show_welcome_screen(username)

    tk.Button(map_root, text="Save & Return", command=save_and_return).pack(pady=20)

    map_root.mainloop()

# ====== Start the app ======
show_login_screen()