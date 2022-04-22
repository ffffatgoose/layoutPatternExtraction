import argparse
from common import utils

def parse_decoder(parser):
    dec_parser = parser.add_argument_group()
    dec_parser.add_argument('--sample_method', type=str,
                        help='"tree" or "radial"')
    dec_parser.add_argument('--motif_dataset', type=str,
                        help='Motif dataset')
    dec_parser.add_argument('--radius', type=int,
                        help='radius of node neighborhoods')
    dec_parser.add_argument('--subgraph_sample_size', type=int,
                        help='number of nodes to take from each neighborhood')
    dec_parser.add_argument('--out_path', type=str,
                        help='path to output candidate motifs')
    dec_parser.add_argument('--n_clusters', type=int)
    dec_parser.add_argument('--min_pattern_size', type=int)
    dec_parser.add_argument('--max_pattern_size', type=int)
    dec_parser.add_argument('--min_neighborhood_size', type=int)
    dec_parser.add_argument('--max_neighborhood_size', type=int)
    dec_parser.add_argument('--n_neighborhoods', type=int)
    dec_parser.add_argument('--n_trials', type=int,
                        help='number of search trials to run')
    dec_parser.add_argument('--out_batch_size', type=int,
                        help='number of motifs to output per graph size')
    dec_parser.add_argument('--analyze', action="store_true")
    dec_parser.add_argument('--search_strategy', type=str,
                        help='"greedy" or "mcts"')
    dec_parser.add_argument('--use_whole_graphs', action="store_true",
        help="whether to cluster whole graphs or sampled node neighborhoods")

    dec_parser.set_defaults(out_path="results/try-layoutGMN-201-205data-test1.p", # result save path
                        n_neighborhoods=10000,
                        n_trials=1000,
                        decode_thresh=0.5,
                        radius=3,
                        subgraph_sample_size=0,
                        sample_method="tree",
                        skip="learnable",
                        min_pattern_size=2,
                        max_pattern_size=8,
                        min_neighborhood_size=3,
                        max_neighborhood_size=15, # random a int between min and max, to random sample emb_graph
                        search_strategy="greedy",
                        out_batch_size=10,
                        use_whole_graphs=True,
                        node_anchored=False)

    parser.set_defaults(dataset="vaulttype2GMN205DECODERposA", # inference dataset name
                        batch_size=1000)
