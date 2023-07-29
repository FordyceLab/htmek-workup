import pymol
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def align_originals(file_list, align_to=0):
    """
    Aligns all structures in file_list to the structure at index align_to.
    """
    pymol.cmd.load(file_list[align_to], "align_to")
    for i, file in enumerate(file_list):
        # if i != align_to:
        pymol.cmd.load(file, "align_from")
        pymol.cmd.align("align_from", "align_to")

        # Save aligned structure
        pymol.cmd.save(file, "align_from")

# define colormapping function
def hex_to_color_format(hex_string):
    # Remove the '#' character if present
    if hex_string.startswith("#"):
        hex_string = hex_string[1:]

    # Convert the hex string to RGB values
    r = int(hex_string[0:2], 16)
    g = int(hex_string[2:4], 16)
    b = int(hex_string[4:6], 16)

    # Create the color format string
    color_format = f"0x40{r:02X}{g:02X}{b:02X}"

    return color_format

def plot_mutational_density(species, design, aligned_pdb_dict, figure_export_path, counts_dict, active_site_dict):
    """
    Plots the mutational density of a design on a structure.

    Parameters:
    species (str): The species of the structure to plot.
    design (str): The design to plot.
    aligned_pdb_dict (dict): A dictionary of aligned PDB filepaths.
    counts_dict (dict): A dictionary of counts for each residue in the design.'

    Returns:
    None
    """

    # clear everything
    pymol.cmd.reinitialize()

    # load structure
    filepath = aligned_pdb_dict[species]
    pdb_id = filepath.split('/')[-1].split('.')[0]

    # load structure
    pymol.cmd.load(filepath)

    # remove water and inorganic molecules
    pymol.cmd.remove("solvent")
    pymol.cmd.remove("inorganic")
    pymol.cmd.remove("resn GOL")

    # set styling
    pymol.cmd.set("shininess", 0)
    pymol.cmd.set("specular", 0.25)
    pymol.cmd.set("ray_trace_mode", 1)
    pymol.cmd.set("ray_trace_fog", 0) # this is important for colormapping, otherwise the colors will be washed out in the background
    pymol.cmd.set("ray_shadow", 0)
    pymol.cmd.set("orthoscopic", 1)

    # rotate 90 degrees in y
    pymol.cmd.rotate("x", 100)

    # set hash max
    pymol.cmd.set("hash_max", 2000)

    # set bg color
    pymol.cmd.bg_color("white")

    # set structure color
    pymol.cmd.color("white")

    # zoom
    pymol.cmd.zoom()

    # get colormap from white to red
    min_val = min(counts_dict.values())
    max_val = max(counts_dict.values())
    colormap = plt.get_cmap('Reds')
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

    # color structure based on hsapiens_progen_counts_dict
    for i, n in counts_dict.items():
        rgba = colormap(norm(n))
        col = hex_to_color_format(mpl.colors.rgb2hex(rgba))
        pymol.cmd.color(col, "resi %s" % (i))
    # create a directory for the images
    if not os.path.exists(figure_export_path + '/pymol_images/design_mutation_density'):
        os.makedirs(figure_export_path + '/pymol_images/design_mutation_density')

    # set the image format
    pymol.cmd.set("ray_trace_mode", 1)

    # create selection of residues
    active_site_residues = [ str(res) for res in active_site_dict[species] ]
    pymol.cmd.select("active_site", "resi " + "+".join(active_site_residues))

    # get all colors
    pymol.cmd.get_color_indices()

    # show the selection as sticks and color blue
    pymol.cmd.show("sticks", "active_site")
    pymol.cmd.color("tv_blue", "active_site")

    # color active site by element, with nitrogen in blue, oxygen in red, and carbon in gray
    pymol.cmd.color("blue", "elem N")
    pymol.cmd.color("red", "elem O")

    # rotate the structure 90 degrees around the y-axis and repeat
    for i in range(0, 4):
        # rotate the structure
        pymol.cmd.rotate("y", 90)

        # ray trace the image
        pymol.cmd.ray(1500)

        # generate the images
        pymol.cmd.png(figure_export_path + '/pymol_images/design_mutation_density/' + species + '/' + species + '_' + design + '_%s.png' % (i), width=1500, height=1500, dpi=300, ray=1)
    # set scaling for size of images
    scale = 3
    plt.rcParams['figure.figsize'] = [scale * 6.0, scale * 5.0]

    # now create a row of subplots and plot the images
    fig, ax = plt.subplots(1, 4)

    for i in range(0, 4):
        # read in and plot the image
        img = plt.imread(figure_export_path + '/pymol_images/design_mutation_density/' + species + '/' + species + '_' + design + '_%s.png' % (i))
        ax[i].imshow(img)
        # add text showing the rotation
        ax[i].text(0.05, 0.95, "Rot = "+str(i * 90) + u"\u00b0", verticalalignment='top', horizontalalignment='left', transform=ax[i].transAxes, fontsize=12)
        # remove the ticks and labels
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])

    # add a title and PDB ID to first plot
    ax[0].set_title(species + ' ' + design.upper(), loc='left', fontsize=12)
    ax[0].text(0, -0.07, 'PDB ID: ' + pdb_id, verticalalignment='top', horizontalalignment='left', transform=ax[0].transAxes, fontsize=12)

    # create a patch with the same color as the active site
    patch = mpl.patches.Rectangle((0, 0), 1, 1, facecolor='blue', edgecolor='black')

    # add colorbar just to the right of the last image
    cax = fig.add_axes([0.92, 0.4, 0.01, 0.2])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=colormap, norm=norm, orientation='vertical')
    cb.set_label('Mutation frequency', rotation=270, labelpad=15)

    # save the figure
    if not os.path.exists(figure_export_path + '/pymol_images/design_mutation_density/' + species + '/'):
        os.makedirs(figure_export_path + '/pymol_images/design_mutation_density/' + species + '/')

    plt.savefig(figure_export_path + '/pymol_images/design_mutation_density/' + species + '/' + species + '_' + design + '_mutation_density.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.show()
    plt.tight_layout()