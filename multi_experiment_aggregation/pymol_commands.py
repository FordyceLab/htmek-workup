import pymol

def align_originals(file_list, align_to=0):
    """
    Aligns all structures in file_list to the structure at index align_to.
    """
    pymol.cmd.load(file_list[align_to])
    for i, file in enumerate(file_list):
        if i != align_to:
            pymol.cmd.load(file)
            pymol.cmd.align(file, file_list[align_to])
            pymol.cmd.save(file, file)
            pymol.cmd.delete(file)