import bin
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Binner for MS2 Spectra')
        parser.add_argument('-mgf', '--mgf', required=True, nargs='+', type=str, help='(Required) Path to .mgf file(s) or directory containing .mgf files')
        parser.add_argument('-b', '--binsize', default=0.01, type=float, help='size of bins for which spectra fall into')
        parser.add_argument('-mb', '--minbin', default=50, type=float, help='minimum bin for the spectra to fall into')
        parser.add_argument('-xb', '--maxbin', default=850, type=float, help='maximum bin for the spectra to fall into')
        parser.add_argument('-p', '--maxmass', default=850, type=float, help='maximum parent mass to filter by')
        parser.add_argument('-f', '--filename', default="binned_data.pkl", type=str, help='filepath to output data to a binary file')
        parser.add_argument('-v', '--verbose', action='store_true', help='turns on verbose mode')
        args = parser.parse_args()

        mgf = args.mgf
        if len(mgf) == 1:
                mgf = mgf[0]

        bin.bin_mgf(mgf_files=mgf, output_file=args.filename, min_bin=args.minbin, max_bin=args.maxbin, bin_size=args.binsize, max_parent_mass=args.maxmass, verbose=args.verbose)