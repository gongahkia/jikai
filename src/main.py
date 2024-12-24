# ----- required imports -----

import agent as a
import helper as h

# ----- sample execution code -----

if __name__ == "__main__":

    TARGET_FILEPATH = "./../corpus/clean/tort/corpus.json"
    LOG_FILEPATH = "./generated_log/log.json"
    TOPICS = ["battery", "assault", "rylands v fletcher"]
    SAMPLE_SIZE = 1

    corpus_data = h.load_corpus(TARGET_FILEPATH)
    if corpus_data is not None:
        context = h.query_relevant_text(corpus_data, TOPICS, 1)
        client = a.start_model()
        if client:
            (agent_role, raw_response) = a.query_hypothetical_generation_model(
                client, context, TOPICS
            )
            h.write_agent_log(LOG_FILEPATH, agent_role, h.sanitise_data(raw_response))
            print("Success: Ok all done")
        else:
            print("Error: Unable to load model")
    else:
        print("Error: Failed to load the corpus.")
