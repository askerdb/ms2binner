import bin
import visualization
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Binner for MS2 Spectra')
        parser.add_argument('-mgf', '--mgf', required=True, nargs='+', type=str, help='(Required) Path to .mgf file(s) or directory containing .mgf files')
        parser.add_argument('-b', '--binsize', default=0.01, type=float, help='size of bins for which spectra fall into')
        parser.add_argument('-mb', '--minbin', default=50, type=float, help='minimum bin for the spectra to fall into')
        parser.add_argument('-xb', '--maxbin', default=850, type=float, help='maximum bin for the spectra to fall into')
        parser.add_argument('-p', '--maxmass', default=850, type=float, help='maximum parent mass to filter by')
        parser.add_argument('-nc', '--components', default=10, type=float, help='number of components to split data into for plotting')
        parser.add_argument('-f', '--filename', default="binned_data.pkl", type=str, help='filepath to output data to a binary file')
        parser.add_argument('-i', '--image', default="spectra_plot", type=str, help='filename to save visualization plot as')
        parser.add_argument('-hl', '--headless', action='store_true', help='turns on headless mode for plotting')
        parser.add_argument('-v', '--verbose', action='store_true', help='turns on verbose mode')
        args = parser.parse_args()

        mgf = args.mgf
        if len(mgf) == 1:
                mgf = mgf[0]

        data, bins, scans = bin.bin_mgf(mgf_files=mgf, output_file=args.filename, min_bin=args.minbin, max_bin=args.maxbin, bin_size=args.binsize, max_parent_mass=args.maxmass, verbose=args.verbose)

        visualization.plot_ms2data(data, num_components=args.components, output_file=args.image, headless=args.headless)