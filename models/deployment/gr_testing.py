import gradio as gr

# Function for mood recommendation
def mood_recommendation(input_text, mood):
    def update(name):
        return f"Welcome to Gradio, {name}!"

    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        with gr.Dropdown(choices=[1 ,2 ,3 ,4]):
            with gr.Row():
                inp = gr.Dropdown(choices=["HAPPY", "SAD"])
                out = gr.Textbox()
            btn = gr.Button("Select")
            btn.click(fn=update, inputs=inp, outputs=out)
    
    return f"Mood recommendation: {input_text}, Mood selected: {mood}"

# Function for similar search
def similar_search(input_text, threshold):
    # Your implementation for similar search here
    return f"Similar search: {input_text}, Threshold: {threshold}"

# Function for clustering playlist
def clustering_playlist(input_text, num_clusters):
    # Your implementation for clustering playlist here
    return f"Clustering playlist: {input_text}, Number of clusters: {num_clusters}"

def choose_function(option, input_text, mood=None, threshold=None, num_clusters=None):
    if option == "Mood_recommendation":
        return mood_recommendation(input_text, mood)
    elif option == "Similar_search":
        return similar_search(input_text, threshold)
    elif option == "Clustering_playlist":
        return clustering_playlist(input_text, num_clusters)
    else:
        return "Invalid option selected"

# Create interface
iface = gr.Interface(
    fn=choose_function,
    inputs=[
        gr.Dropdown(choices=["Mood_recommendation", "Similar_search", "Clustering_playlist"], label="Choose an option"),
        gr.Textbox(label="Enter input")
    ],
    outputs=gr.Textbox(label="Output")
)



iface.launch()
