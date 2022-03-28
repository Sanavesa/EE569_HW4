import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# parses the descriptor file into a numpy array
def parse_descriptor_file(filename):
    with open(f"{filename}_descriptors.txt", "r") as f:
        lines = f.readlines()
        
        # 4th line has #rows
        rows = [int(s) for s in lines[3].split() if s.isdigit()][0]
        
        # 5th line has #cols
        cols = [int(s) for s in lines[4].split() if s.isdigit()][0]
        
        # 6th line onwards has the data 
        data = ""
        for line in lines[6:]:
            data += line
            
        # Parse the [...] data to float array
        openBracket = data.find("[")
        closeBracket = data.find("]")
        data = data[openBracket+1:closeBracket].strip()
        data = list(map(float, data.split(",")))

        values = np.array(data, dtype=np.float32).reshape(rows, cols)
        print(f"Loaded {filename} {rows} descriptors")
        return values

# Computes the normalized histogram
def calc_hist(bow_features):
    counts = np.bincount(bow_features)
    print(counts)
    return counts / np.sum(counts)

# Get the similarity via histogram intersection method
def calc_similarity(hist1, hist2):
    min_hist = np.minimum(hist1, hist2) # elementwise min
    max_hist = np.maximum(hist1, hist2) # elementwise max
    return min_hist.sum() / max_hist.sum()

# plots the histograms into one figure
def plot_hists(filename, hists, labels, colors, title, xlabel, ylabel):
    fig = plt.figure()
    xvalues = range(1, 9)
    bar_width = 0.25
       
    last_r = np.arange(len(hists[0]))
    for (hist, label, color) in zip(hists, labels, colors):
        plt.bar(last_r, hist, color=color, width=bar_width, edgecolor='white', label=label)
        last_r = [x + bar_width for x in last_r]
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([r + bar_width for r in range(len(hist))], xvalues)
    
    plt.legend()
    plt.savefig(filename)
    # plt.show()

cat_1_descriptor = parse_descriptor_file("build/debug/cat_1")
cat_2_descriptor = parse_descriptor_file("build/debug/cat_2")
dog_1_descriptor = parse_descriptor_file("build/debug/dog_1")
cat_dog_descriptor = parse_descriptor_file("build/debug/cat_dog")
dog_2_descriptor = parse_descriptor_file("build/debug/dog_2")

for n_components in [-1, 20]:
    print("-"*100)
    print("Number of components", n_components)
    if n_components == -1: # no pca
        cat_1_descriptor_pca = cat_1_descriptor
        cat_2_descriptor_pca = cat_2_descriptor
        dog_1_descriptor_pca = dog_1_descriptor
        cat_dog_descriptor_pca = cat_dog_descriptor
        dog_2_descriptor_pca = dog_2_descriptor
    else:
        cat_1_descriptor_pca = PCA(n_components=n_components).fit_transform(cat_1_descriptor)
        cat_2_descriptor_pca = PCA(n_components=n_components).fit_transform(cat_2_descriptor)
        dog_1_descriptor_pca = PCA(n_components=n_components).fit_transform(dog_1_descriptor)
        cat_dog_descriptor_pca = PCA(n_components=n_components).fit_transform(cat_dog_descriptor)
        dog_2_descriptor_pca = PCA(n_components=n_components).fit_transform(dog_2_descriptor)
        
    x_train = np.vstack([cat_1_descriptor_pca, cat_2_descriptor_pca, dog_1_descriptor_pca, cat_dog_descriptor_pca])
    kmeans = KMeans(n_clusters=8, init="k-means++").fit(x_train)

    cat_1_bow = kmeans.predict(cat_1_descriptor_pca)
    cat_1_hist = calc_hist(cat_1_bow)

    dog_1_bow = kmeans.predict(dog_1_descriptor_pca)
    dog_1_hist = calc_hist(dog_1_bow)

    dog_2_bow = kmeans.predict(dog_2_descriptor_pca)
    dog_2_hist = calc_hist(dog_2_bow)

    print(f"cat_1 and dog_2 similarity = {calc_similarity(cat_1_hist, dog_2_hist):.2f}")
    print(f"dog_1 and dog_2 similarity = {calc_similarity(dog_1_hist, dog_2_hist):.2f}")
    plot_hists(
        filename=f"hists_{n_components}.png",
        hists=[cat_1_hist, dog_1_hist, dog_2_hist],
        labels=["cat_1", "dog_1", "dog_2"],
        colors=["red", "green", "blue"],
        title="Normalized Histogram for each Image after K-means",
        xlabel="Codeword",
        ylabel="Normalized Count")