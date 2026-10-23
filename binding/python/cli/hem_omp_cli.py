"""OMP command grammar and fixed integration dispatch."""
from pathlib import Path

from . import hem_arguments as arguments
from ..hem_integrations import configure_integration


def configure(parser):
    commands=parser.add_subparsers(dest='command',required=True)
    child=commands.add_parser('configure',help='configure or inspect Oh My Pi integration')
    arguments.output(child);arguments.lifecycle(child)
    mode=child.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='inspect without writes or client execution')
    mode.add_argument('--disable',action='store_true',help='remove unchanged owned entries without client')
    child.add_argument('--omp-dir',type=Path,help='explicit default-profile agent directory')


def run(args,deadline):
    mode='disable' if args.disable else 'check' if args.check else 'configure'
    return configure_integration('omp',None,deadline,mode=mode,omp_dir=args.omp_dir)
