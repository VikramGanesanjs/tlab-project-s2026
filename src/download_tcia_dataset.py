from idc_index import IDCClient
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--collection_id', type=str, required=True)
parser.add_argument('--download_dir', type=str, required=True)
args = parser.parse_args()

idc_client = IDCClient()

idc_client.download_from_selection(collection_id=args.collection_id, downloadDir=args.download_dir)