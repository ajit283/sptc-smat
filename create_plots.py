import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


COLS = 8

NONE = "base"
TWICE = "row+column"
ONCE = "row"

def parse_dasp_file(filename, ordering=NONE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    nnz_map = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        if len(dubrovnik) == 7:
            # cusparse
            name = dubrovnik[0].split('/')[-1].split('.')[0]
            time = float(dubrovnik[5])
            nnz = int(dubrovnik[3])
            nnz_map[name] = nnz
            res["name"].append(name)
            res["time"].append(time)
            res["algo"].append("cuSPARSE")
            res["ordering"].append(ordering)

        else:
            # dasp
            name = dubrovnik[0].split('/')[-1].split('.')[0]
            time = float(dubrovnik[-7]) * COLS
            res["name"].append(name)
            res["time"].append(time)
            res["algo"].append("DASP")
            res["ordering"].append(ordering)

    return res, nnz_map


def parse_dasp_file_ordered(filename):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    nnz_map = {}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        if len(dubrovnik) == 7:
            # cusparse
            name = dubrovnik[0].split('/')[-1].split('.')[0]
            index = name.find('_reordered')
            name = name[:index]
            time = float(dubrovnik[5])
            nnz = int(dubrovnik[3])
            nnz_map[name] = nnz
            res["name"].append(name)
            res["time"].append(time)
            res["algo"].append("cuSPARSE")
            
        else:
            # dasp
            name = dubrovnik[0].split('/')[-1].split('.')[0]
            index = name.find('_reordered')
            name = name[:index]
            time = float(dubrovnik[-7]) * COLS
            res["name"].append(name)
            res["time"].append(time)
            res["algo"].append("DASP")
            
        if "transposed" in dubrovnik[0]:
            res["ordering"].append(TWICE)
        else:
            res["ordering"].append(ONCE)
    return res, nnz_map

def parse_smatel(filename, ordering=TWICE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        # SMaTeL
        name = dubrovnik[0].split('/')[-1].split('.')[0]
        last_underscore_index = name.find('_reordered')
        name = name[:last_underscore_index]
        time = float(dubrovnik[1])
        res["name"].append(name)
        res["time"].append(time)
        res["algo"].append("SMaTeL")
        res["ordering"].append(ordering)
    return res

def parse_smatel_all(filename):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        # SMaTeL
        name = dubrovnik[0].split('/')[-1].split('.')[0]
        if "reordered" in name:
            last_underscore_index = name.find('_reordered')
            if "transposed" in name:
                res["ordering"].append(TWICE)
            else:
                res["ordering"].append(ONCE)
            name = name[:last_underscore_index]
        else:
            res["ordering"].append(NONE)
        
        time = float(dubrovnik[1])
        res["name"].append(name)
        res["time"].append(time)
        res["algo"].append("SMaT")
        
    return res

def parse_smatel_once(filename, ordering=ONCE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        # SMaTeL
        name = dubrovnik[0].split('/')[-1].split('.')[0]
        last_underscore_index = name.find('_reordered')
        name = name[:last_underscore_index]
        time = float(dubrovnik[1])
        res["name"].append(name)
        res["time"].append(time)
        res["algo"].append("SMaTeL")
        res["ordering"].append(ordering)

    return res

def parse_smatel_no(filename, ordering=NONE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        dubrovnik = line.split(",")
        # SMaTeL
        name = dubrovnik[0].split('/')[-1].split('.')[0]
        time = float(dubrovnik[1])
        res["name"].append(name)
        res["time"].append(time)
        res["algo"].append("SMaTeL")
        res["ordering"].append(ordering)

    return res

def parse_magicube_file(filename, ordering=NONE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('Sparse'):
            name = line.split()[-1].split('/')[-1].split('.')[0]
            last_underscore_index = name.rfind('_')
            name = name[:last_underscore_index]
            res["name"].append(name)
            res["algo"].append("Magicube")
            res["ordering"].append(ordering)
        elif line.startswith('Magicube'):
            time = float(line.split()[-2])
            res["time"].append(time)

    return res

def parse_magicube_file_ordered(filename, ordering=NONE):
    res = {"name": [], "time": [], "algo": [], "ordering": []}
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith('Sparse'):
            name = line.split()[-1].split('/')[-1].split('.')[0]
            if "transposed" in name:
                res["ordering"].append(TWICE)
            else:
                res["ordering"].append(ONCE)
            last_underscore_index = name.find('_reordered')
            name = name[:last_underscore_index]
            res["name"].append(name)
            res["algo"].append("Magicube")
        
        elif line.startswith('Magicube'):
            time = float(line.split()[-2])
            res["time"].append(time)

    return res
        

def create_bar_chart():
    # Define the data for the bars
    x = ['Group 1', 'Group 2', 'Group 3']
    y1 = [10, 15, 12]  # Values for the first group
    y2 = [8, 11, 9]    # Values for the second group

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    bar_positions1 = range(len(x))
    bar_positions2 = [pos + bar_width for pos in bar_positions1]

    # Create the bar chart
    plt.bar(bar_positions1, y1, width=bar_width, label='Group 1')
    plt.bar(bar_positions2, y2, width=bar_width, label='Group 2')

    # Add labels and title
    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.title('Grouped Bar Chart')

    # Add legend
    plt.legend()

    # Show the chart
    plt.show()

def grouped_bar_chart():
    species = ("Adelie", "Chinstrap", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.show()


def grouped_bar_chart_tflops():
    species = ("Adelie", "Chinstrap", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.show()
        

def time_depending_on_n():

    # Create a figure and axes
    fig, ax = plt.subplots()
    

    # DASP
    x = [1, 4, 8, 16, 24, 32,] # 40, 48, 56, 64, 72, 80, 88, 96, 104] #200, 400, 1000]
    y = [0.0297 * i for i in x]

    # Create the plot
    plt.plot(x, y, label='DASP', marker='x', color='tab:green')

    
    # plt.title('DASP Performance')

    # cusparse
    x = [1, 4, 8, 16, 24, 32, ] #40, 48, 56, 64, 72, 80, 88, 96, 104]# 200, 400, 1000]
    y = [0.293400, 0.457800, 0.705500, 1.142200, 1.824100, 2.280200,] # 3.096400, 3.522400, 4.299400, 4.725100, 4.410700, 4.743300, 6.979200, 7.160500, 7.834200] # 14.315900, 24.538700, 60.067200]
    plt.plot(x, y, label='cuSPARSE', marker='o', color='tab:orange')

    # Magicube
    x = [1, 4, 8, 16, 24, 32,] # 40, 48, 56, 64, 72, 80, 88, 96, 104] #200, 400, 1000]
    y = [0.744876, 0.766356, 0.777793, 0.816469, 0.815408, 0.827263,] # 0.783243, 0.775415, 0.788676, 0.840527, 1.54783, 1.54311, 1.54301, 1.52666, 1.60832] # 3.06065, 5.3284, 12.1257]
    plt.plot(x, y, label='Magicube', marker='s', color='tab:blue')

    # ours
    # List of values for the stairs
    val = [0.125542, 0.179610, 0.239616, 0.298086, 0.295731,] # 0.343142, 0.391578, 0.439808, 0.485334, 0.531610, 0.579130, 0.626765, 0.720915]
    x, y = [], []
    found = False
    for i in range(len(val)):
        for j in range(1, 9):
            if i * 8 + j == 32:
                found = True
            x.append(i * 8 + j)
            y.append(val[i])
        if found:
            break
    # Create the stairs plot
    print(x, y)
    plt.step(x, y, where='post', label='SMaT', color='tab:red')

    plt.grid(False)
    ax.tick_params(color='black', labelcolor='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    plt.xlabel('N', fontsize=12, ) #fontweight='bold')
    plt.ylabel('Compute time [ms]', fontsize=12,) # fontweight='bold')
    # Add legend
    plt.legend(fontsize = 12)

    ax.grid(axis='y')

    plt.savefig('zoomedin_lineplot.pdf')


    # Show the plot
    plt.show()
    


def plot_bar_chart(df, filename="representative.pdf"):
    plt.figure(figsize=(20, 10))  # Adjust the width as needed
    sns.barplot(df, x="name", y="GFLOPS", hue="algo")
    plt.grid(False)

    plt.ylabel('GFLOPS', fontsize=12, fontweight='bold')
    plt.xlabel("")
    plt.legend(title='',)
    plt.savefig(filename)    


df1 = pd.DataFrame(parse_magicube_file("magicube.out"))
res, nnz_map = parse_dasp_file("spmv_f16_record_2103.csv")
df2 = pd.DataFrame(res)
df3 = pd.DataFrame(parse_smatel("twice_ordered.csv"))
possible_smatel = df3["name"].unique()
df = pd.concat([df1, df2, df3], ignore_index=True)
df["nnz"] = df["name"].map(nnz_map)
df["GFLOPS"] = (2 * df["nnz"] * COLS) / (df["time"] * 1e6)
df = df[df['name'].isin(possible_smatel)]

orderA = ['mip1', 'cant', 'pdb1HYS', 'rma10', 'cop20k_A', 'consph', 'shipsec1','dc2', 'conf5_4..']
hue_orderA = [NONE, ONCE, TWICE]

# smatel - rerun
df_smatel = pd.DataFrame(parse_smatel_all("representative_orderings_smatel.csv"))
df_smatel["nnz"] = df_smatel["name"].map(nnz_map)
df_smatel["GFLOPS"] = (2 * df_smatel["nnz"] * COLS) / (df_smatel["time"] * 1e6)
df_smatel['name'] = df_smatel['name'].str.replace('conf5_4-8x8-10', 'conf5_4..')

plt.figure(figsize=(20, 10))  # Adjust the width as needed
sns.barplot(df_smatel, x="name", y="GFLOPS", hue="ordering", order=orderA, hue_order=hue_orderA)
plt.grid(False)

fs = 26
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Performance [GFLOP/s]', fontsize=fs, ) # fontweight='bold')
plt.xlabel("")
plt.legend(title='', fontsize=fs)
plt.savefig("ordering_effect_smatel.pdf")


# dasp
res1, nnz_map = parse_dasp_file("spmv_f16_record_2103.csv")
df1_dasp = pd.DataFrame(res1)
res2, nnz_map = parse_dasp_file_ordered("spmm_twice_once_oredered_cusparse_dasp.csv")
df2_dasp = pd.DataFrame(res2)
df_dasp = pd.concat([df1_dasp, df2_dasp,], ignore_index=True)
df_dasp = df_dasp[df_dasp['name'].isin(possible_smatel)]
df_dasp = df_dasp[df_dasp['algo'] == "DASP"]
df_dasp["nnz"] = df_dasp["name"].map(nnz_map)
df_dasp["GFLOPS"] = (2 * df_dasp["nnz"] * COLS) / (df_dasp["time"] * 1e6)
df_dasp['name'] = df_dasp['name'].str.replace('conf5_4-8x8-10', 'conf5_4..')
plt.figure(figsize=(20, 10))  # Adjust the width as needed
sns.barplot(df_dasp, x="name", y="GFLOPS", hue="ordering", order=orderA, hue_order=hue_orderA)
plt.grid(False)


fs = 26
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Performance [GFLOP/s]', fontsize=fs, ) # fontweight='bold')
plt.xlabel("")
plt.legend(title='', fontsize=fs)
plt.savefig("ordering_effect_dasp.pdf")


# magicube
df1_magicube = pd.DataFrame(parse_magicube_file("magicube.out"))
df2_magicube = pd.DataFrame(parse_magicube_file_ordered("representative_ordered_magicube.out"))

df_magicube = pd.concat([df1_magicube, df2_magicube,], ignore_index=True)
df_magicube = df_magicube[df_magicube['name'].isin(possible_smatel)]
print(df_magicube)
df_magicube = df_magicube[df_magicube['algo'] == "Magicube"]
df_magicube["nnz"] = df_magicube["name"].map(nnz_map)
df_magicube["GFLOPS"] = (2 * df_magicube["nnz"] * COLS) / (df_magicube["time"] * 1e6)
df_magicube['name'] = df_magicube['name'].str.replace('conf5_4-8x8-10', 'conf5_4..')

plt.figure(figsize=(20, 10))  # Adjust the width as needed
sns.barplot(df_magicube, x="name", y="GFLOPS", hue="ordering", order=orderA, hue_order=hue_orderA)
plt.grid(False)

fs = 26
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Performance [GFLOP/s]', fontsize=fs, ) # fontweight='bold')
plt.xlabel("")
plt.legend(title='', fontsize=fs)
plt.savefig("ordering_effect_magicube.pdf")


# cusparse
res1, nnz_map = parse_dasp_file("spmv_f16_record_2103.csv")
df1_cusparse = pd.DataFrame(res1)
res2, nnz_map = parse_dasp_file_ordered("spmm_twice_once_oredered_cusparse_dasp.csv")
df2_cusparse = pd.DataFrame(res2)
df_cusparse = pd.concat([df1_cusparse, df2_cusparse,], ignore_index=True)
df_cusparse = df_cusparse[df_cusparse['name'].isin(possible_smatel)]
df_cusparse = df_cusparse[df_cusparse['algo'] == "cuSPARSE"]
df_cusparse["nnz"] = df_cusparse["name"].map(nnz_map)
df_cusparse["GFLOPS"] = (2 * df_cusparse["nnz"] * COLS) / (df_cusparse["time"] * 1e6)
df_cusparse['name'] = df_cusparse['name'].str.replace('conf5_4-8x8-10', 'conf5_4..')

plt.figure(figsize=(20, 10))  # Adjust the width as needed
sns.barplot(df_cusparse, x="name", y="GFLOPS", hue="ordering", order=orderA, hue_order=hue_orderA)
plt.grid(False)

fs = 26
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Performance [GFLOP/s]', fontsize=fs, ) # fontweight='bold')
plt.xlabel("")
plt.legend(title='', fontsize=fs)
plt.savefig("ordering_effect_cusparse.pdf")


res, nnz_map = parse_dasp_file("spmv_f16_record_2103.csv")
df2 = pd.DataFrame(res)
df2["nnz"] = df2["name"].map(nnz_map)
df2["GFLOPS"] = (2 * df2["nnz"] * COLS) / (df2["time"] * 1e6)

df1 = pd.DataFrame(parse_magicube_file("magicube.out"))
df1["nnz"] = df1["name"].map(nnz_map)
df1["GFLOPS"] = (2 * df1["nnz"] * COLS) / (df1["time"] * 1e6)


df3 = pd.DataFrame(parse_smatel_all("representative_orderings_smatel.csv"))
df3["nnz"] = df3["name"].map(nnz_map)
df3["GFLOPS"] = (2 * df3["nnz"] * COLS) / (df3["time"] * 1e6)
max_indices = df3.groupby('name')['GFLOPS'].idxmax()
df3 = df3.loc[max_indices]

possible_smatel = df3["name"].unique()
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df[df['name'].isin(possible_smatel)]



df["TFLOPS"] = (df["GFLOPS"] / 1000.)
print(df["name"].unique())
print(df[df["algo"] == "SMaT"])
plt.figure(figsize=(20, 5))  # Adjust the width as needed
ord = ['mip1', 'cant', 'pdb1HYS', 'rma10', 'cop20k_A', 'consph', 'shipsec1','dc2', 'conf5_4-8x8-10']
sns.barplot(df, x="name", y="GFLOPS", hue="algo", order=ord)
plt.grid(False)

fs = 14
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('Performance [GFLOP/s]', fontsize=fs, ) # fontweight='bold')
plt.xlabel("")
plt.legend(title='', fontsize=fs)
plt.savefig("representative2.svg")

time_depending_on_n()


fig, axes = plt.subplots(1, 9, figsize=(30, 5))

# Flatten the axes array to simplify indexing
axes = axes.flatten()
for idx, (mat, ax) in enumerate(zip(dfnew["matrix"].unique(), axes)):
    dfic = dfnew[dfnew["matrix"] == mat]

    sns.violinplot(data=dfic, x='matrix', y='blocks', hue="variant", ax=ax, legend=False)
    
    if mat == "dc2" or mat == "mip1":
        current_y_ticks = ax.get_yticks()
        current_y_ticks = [np.exp(y) for y in current_y_ticks]
        current_y_ticks = [f"{int(y)}" for y in current_y_ticks]
        print(current_y_ticks)
        ax.set_yticklabels(current_y_ticks)

    fs = 26
    ax.grid(axis='y')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylabel("", fontsize=fs, ) # fontweight='bold')
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("violing9x1.svg")
# Show the plot
plt.show()