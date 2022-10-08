import matplotlib.pyplot as plt

RGBAColor = tuple[float, float, float, float]

def get_index_colors() -> list[RGBAColor]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]