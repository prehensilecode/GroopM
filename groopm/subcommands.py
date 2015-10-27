#!/usr/bin/env python
###############################################################################
#                                                                             #
#    subcommands.py                                                           #
#                                                                             #
#    Command line programs                                                    #
#                                                                             #
#    Copyright (C) Tim Lamberton, Michael Imelfort                            #
#                                                                             #
###############################################################################
#                                                                             #
#          .d8888b.                                    888b     d888          #
#         d88P  Y88b                                   8888b   d8888          #
#         888    888                                   88888b.d88888          #
#         888        888d888 .d88b.   .d88b.  88888b.  888Y88888P888          #
#         888  88888 888P"  d88""88b d88""88b 888 "88b 888 Y888P 888          #
#         888    888 888    888  888 888  888 888  888 888  Y8P  888          #
#         Y88b  d88P 888    Y88..88P Y88..88P 888 d88P 888   "   888          #
#          "Y8888P88 888     "Y88P"   "Y88P"  88888P"  888       888          #
#                                             888                             #
#                                             888                             #
#                                             888                             #
#                                                                             #
###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

__author__ = "Michael Imelfort, Tim Lamberton"
__copyright__ = "Copyright 2012-2015"
__credits__ = ["Tim Lamberton", "Michael Imelfort"]
__license__ = "GPL3"
__maintainer__ = "Tim Lamberton"
__email__ = "t.lamberton@uq.edu.au"

###############################################################################

import argparse
import groopm
from groopmExceptions import ExtractModeNotAppropriateException
from .version import __version__

###############################################################################
###############################################################################
###############################################################################
###############################################################################

#------------------------------------------------------------------------------
#Helpers

class CustomHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        h = action.help
        if '%(default)' not in action.help:
            if action.default != '' and \
               action.default != [] and \
               action.default != None \
               and action.default != False:
                if action.default is not argparse.SUPPRESS:
                    defaulting_nargs = [argparse.OPTIONAL,
                                        argparse.ZERO_OR_MORE]

                    if action.option_strings or action.nargs in defaulting_nargs:

                        if '\n' in h:
                            lines = h.splitlines()
                            lines[0] += ' (default: %(default)s)'
                            h = '\n'.join(lines)
                        else:
                            h += ' (default: %(default)s)'
        return h

    def _fill_text(self, text, width, indent):
        return ''.join([indent + line for line in text.splitlines(True)])


###############################################################################
# Workflow
###############################################################################

#------------------------------------------------------------------------------
#Parse

class ParseSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("parse",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="parse raw data and save to disk",
                                       description="Parse raw data and save to disk")
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of the database being created")
        parser.add_argument('reference', help="fasta file containing bam reference sequences")
        parser.add_argument('bamfiles', nargs='+', help="bam files to parse")
        parser.add_argument('-t', '--threads', type=int, default=1, help="number of threads to use during BAM parsing")
        parser.add_argument('-f', '--force', action="store_true", default=False, help="overwrite existing DB file without prompting")
        parser.add_argument('-c', '--cutoff', type=int, default=500, help="cutoff contig size during parsing")

    def parse_options(self, options):
        timer = groopm.TimeKeeper()
        print "*******************************************************************************"
        print " [[GroopM %s]] Running in data parsing mode..." % __version__
        print "*******************************************************************************"
        # check this here:
        if len(options.bamfiles) < 3:
            print "Sorry, You must supply at least 3 bamFiles to use GroopM. (You supplied %d)\n Exiting..." % len(options.bamfiles)
            return
        dm = groopm.DataManager()
        success = dm.createDB(options.bamfiles,
                                  options.reference,
                                  options.dbname,
                                  options.cutoff,
                                  timer,
                                  force=options.force,
                                  threads=options.threads)
        if not success:
            print options.dbname,"not updated"

#------------------------------------------------------------------------------
#Core

class CoreSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("core",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="load saved data and make bin cores",
                                       description="Load saved data and make bin cores")
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of the database to open")
        parser.add_argument('-c', '--cutoff', type=int, default=1500, help="cutoff contig size for core creation")
        parser.add_argument('-s', '--size', type=int, default=10, help="minimum number of contigs which define a core")
        parser.add_argument('-b', '--bp', type=int, default=1000000, help="cumulative size of contigs which define a core regardless of number of contigs")
        parser.add_argument('-f', '--force', action="store_true", default=False, help="overwrite existing DB file without prompting")

    def parse_options(self, options):
        timer = groopm.TimeKeeper()
        print "*******************************************************************************"
        print " [[GroopM %s]] Running in core creation mode..." % __version__
        print "*******************************************************************************"

        ce = groopm.ClusterEngine(options.dbname, minSize=options.size, minBP=options.bp)
        ce.run(timer, minLength=options.cutoff, force=options.force)


#------------------------------------------------------------------------------
#Extract

class ExtractSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("extract",
                                       formatter_class=CustomHelpFormatter,
                                       help='extract contigs or reads based on bin affiliations',
                                       description="Extract contigs or reads based on bin affiliations",
                                       epilog='''Example usage:

 Extract contigs from bin 33:

  $ groopm extract my_db.gm my_contigs.fasta --bids 33

 Extract reads mapping to contigs in bin 35:

  $ groopm extract my_db.gm my.bam -bids 35 --mode reads

''')
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of the database to open")
        parser.add_argument('data', nargs='+', help="data file(s) to extract from, bam or fasta")
        parser.add_argument('-b', '--bids', nargs='+', type=int, default=None, help="bin ids to use (None for all)")
        parser.add_argument('-m', '--mode', default="contigs", help="what to extract", choices=('contigs','reads'))
        parser.add_argument('-o', '--out_folder', default="", help="write to this folder (None for current dir)")
        parser.add_argument('-p', '--prefix', default="", help="prefix to apply to output files")

        contig_extraction_options=parser.add_argument_group('Contigs mode extraction options')
        contig_extraction_options.add_argument('-c', '--cutoff', type=int, default=0, help="cutoff contig size (0 for no cutoff)")

        read_extraction_options=parser.add_argument_group('Reads mode extraction options')
        read_extraction_options.add_argument('--mix_bams', action="store_true", default=False, help="use the same file for multiple bam files")
        read_extraction_options.add_argument('--mix_groups', action="store_true", default=False, help="use the same files for multiple group groups")
        read_extraction_options.add_argument('--mix_reads', action="store_true", default=False, help="use the same files for paired/unpaired reads")
        read_extraction_options.add_argument('--interleave', action="store_true", default=False, help="interleave paired reads in ouput files")
        read_extraction_options.add_argument('--headers_only', action="store_true", default=False, help="extract only (unique) headers")
        read_extraction_options.add_argument('--no_gzip', action="store_true", default=False, help="do not gzip output files")

        read_extraction_options.add_argument('--mapping_quality', type=int, default=0, help="mapping quality threshold")
        read_extraction_options.add_argument('--use_secondary', action="store_true", default=False, help="use reads marked with the secondary flag")
        read_extraction_options.add_argument('--use_supplementary', action="store_true", default=False, help="use reads marked with the supplementary flag")
        read_extraction_options.add_argument('--max_distance', type=int, default=1000, help="maximum allowable edit distance from query to reference")

        read_extraction_options.add_argument('-v', '--verbose', action="store_true", default=False, help="be verbose")
        read_extraction_options.add_argument('-t', '--threads', type=int, default=1, help="maximum number of threads to use")

        return parser

    def parse_options(self, options):
        timer = groopm.TimeKeeper()
        print "*******************************************************************************"
        print " [[GroopM %s]] Running in '%s' extraction mode..." % (__version__, options.mode)
        print "*******************************************************************************"
        bids = []
        if options.bids is not None:
            bids = options.bids
        bx = groopm.BinExtractor(options.dbname,
                                 bids=bids,
                                 folder=options.out_folder
                                )
        if(options.mode=='contigs'):
            bx.extractContigs(timer,
                              fasta=options.data,
                              prefix=options.prefix,
                              cutoff=options.cutoff)

        elif(options.mode=='reads'):
            bx.extractReads(timer,
                            bams=options.data,
                            prefix=options.prefix,
                            mixBams=options.mix_bams,
                            mixGroups=options.mix_groups,
                            mixReads=options.mix_reads,
                            interleaved=options.interleave,
                            bigFile=options.no_gzip,
                            headersOnly=options.headers_only,
                            minMapQual=options.mapping_quality,
                            maxMisMatches=options.max_distance,
                            useSuppAlignments=options.use_supplementary,
                            useSecondaryAlignments=options.use_secondary,
                            verbose=options.verbose,
                            threads=options.threads)

        else:
            raise ExtractModeNotAppropriateException("mode: "+ options.mode + " is unknown")

###############################################################################
# Import Export
###############################################################################

#------------------------------------------------------------------------------
#Dump

class DumpSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("dump",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="write database to text file",
                                       description="Write database to text file")
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of the database to open")
        parser.add_argument('-f', '--fields', default="names,bins", help="fields to extract: Build a comma separated list from [names, mers, gc, coverage, tcoverage, ncoverage, lengths, bins] or just use 'all'")
        parser.add_argument('-o', '--outfile', default="GMdump.csv", help="write data to this file")
        parser.add_argument('-s', '--separator', default=",", help="data separator")
        parser.add_argument('--no_headers', action="store_true", default=False, help="don't add headers")

    def parse_options(self, options):
        timer = groopm.TimeKeeper()
        print "*******************************************************************************"
        print " [[GroopM %s]] Running in data dumping mode..." % __version__
        print "*******************************************************************************"

        # prep fields. Do this first cause users are mot likely to
        # mess this part up!
        allowable_fields = ['names', 'mers', 'gc', 'coverage', 'tcoverage', 'ncoverage', 'lengths', 'bins', 'all']
        fields = options.fields.split(',')
        for field in fields:
            if field not in allowable_fields:
                print "ERROR: field '%s' not recognised. Allowable fields are:" % field
                print '\t',",".join(allowable_fields)
                return
        if options.separator == '\\t':
            separator = '\t'
        else:
            separator = options.separator

        dm = groopm.DataManager()
        dm.dumpData(options.dbname,
                    fields,
                    options.outfile,
                    separator,
                    not options.no_headers)


#------------------------------------------------------------------------------
#Import

class ImportSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("import",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="import information from text file",
                                       description="Import information from text file")
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of the database to open")
        parser.add_argument('infile', help="file with data to import")
        parser.add_argument('-t', '--fields', default="bins", help="data type to import. [bins]")
        parser.add_argument('-s', '--separator', default=",", help="data separator")
        parser.add_argument('--has_headers', action="store_true", default=False, help="file contains headers")

        return parser

    def parse_options(self, options):
        pass

###############################################################################
# Plotting
###############################################################################

#------------------------------------------------------------------------------
# Dump

class PlotSubcommand:
    def add_subparser_to(self, subparsers):
        parser = subparsers.add_parser("plot",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                       help="plot contigs in coverage vs kmer distance space",
                                       description="Plot contigs in coverage vs kmer distance space")
        self.add_arguments_to(parser)
        parser.set_defaults(parse=self.parse_options)
        return parser

    def add_arguments_to(self, parser):
        parser.add_argument('dbname', help="name of database to open")
        parser.add_argument('-b', '--bids', nargs='*', type=int, default=None, help="bin ids to plot (None for all)")
        #parser.add_argument('--names', nargs='*', default=None, help="contig ids to plot")
        parser.add_argument('-p', '--prefix', default="BIN", help="prefix to apply to output files")
        parser.add_argument('-o', '--out_folder', default="", help="save plots in folder")
        parser.add_argument('-i', '--interactive', action="store_true", default=False, help="interatcive plot first bin or contig id")
        parser.add_argument('--ranks', action="store_true", default=False, help="plot variable ranks")
        parser.add_argument('--origin', default="mediod", choices=["mediod", "max_length", "max_coverage"], help="set how to choose bin centers")
        parser.add_argument('--highlight', default="cluster", choices=["cluster", "mergers"], help="choose how to highlight contigs")
        parser.add_argument('--colormap', default="HSV", choices=["HSV", "Accent", "Blues", "Spectral", "Grayscale", "Discrete", "DiscretePaired"], help="set colormap")

        return parser

    def parse_options(self, options):
        timer = groopm.TimeKeeper()
        print "*******************************************************************************"
        print " [[GroopM %s]] Running in plotting mode..." % __version__
        print "*******************************************************************************"


        bplot = groopm.BinPlotter(options.dbname,
                                  folder=None if options.interactive else options.out_folder)
        bplot.plot(timer,
                   bids=options.bids,
                   origin_mode=options.origin,
                   highlight_mode=options.highlight,
                   threshold=0.5,
                   plotRanks=options.ranks,
                   colorMap=options.colormap,
                   prefix=options.prefix)


###############################################################################
###############################################################################
###############################################################################
###############################################################################

