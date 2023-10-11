from html import entities
import scispacy
from scispacy.linking import EntityLinker
import spacy
import fire
import json
import os


def main(data, mode = "scispacy"):
    srcs = [json.loads(d)['src'] for d in open(f"output/preprocessed/{data}/test.json").readlines()]
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    linker = nlp.get_pipe("scispacy_linker")
    docs = [nlp(text) for text in srcs]
    results = []
    for d in docs:
        res = []
        for e in d.ents:
            for umls_ent in e._.kb_ents:
                concept = linker.kb.cui_to_entity[umls_ent[0]].canonical_name
                res.append(f"{e} [SEP] {concept} [EOS] ")
                if len(res) >= 5:
                    break
        results.append(''.join(res))
    os.system(f"mkdir -p output/exps/{data}__{mode}")
    open(f"output/exps/{data}__{mode}/generated_predictions.txt", "w").write("\n".join(results))

if __name__ == "__main__":
    #main("cometa_stratified")
    fire.Fire(main)