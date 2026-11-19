# Demo examples

`cases.json` holds ten investigator trajectories exported from the released WeVisForensics-DTPO model, with the matching videos and posters in this folder. They cover every inspector type and include one real video. Each step has the investigator's reasoning, the called inspector with its window, and the inspector report; the last step holds the final answer.

`demo/app.py` shows them in its example gallery; with `WVF_LOCAL=0` it renders them without a GPU.
