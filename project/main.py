import asyncio

import ingest
import logs
import agent_building


REPO_OWNER = "chiphuyen"
REPO_NAME = "aie-book"


def initialize_index():
    print(f"Starting AI Engineering Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion...")

    index = ingest.index_data(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        chunk=True,
        chunking_strategy="section",
        chunking_params={"level": 2},
    )

    print("Data indexing completed successfully!")
    return index


def initialize_agent(index):
    print("Initializing search agent...")

    agent = agent_building.init_agent(
        index=index,
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
    )

    print("Agent initialized successfully!")
    return agent


def main():
    index = initialize_index()
    agent = initialize_agent(index)

    system_prompt = agent_building.SYSTEM_PROMPT_TEMPLATE.format(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
    )

    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")

        if question.strip().lower() == "stop":
            print("Goodbye!")
            break

        print("Processing your question...")

        response = asyncio.run(agent.run(user_prompt=question))

        logs.log_interaction_to_file(
            agent=agent,
            messages=response.new_messages(),
            system_prompt=system_prompt,
        )

        print("\nResponse:\n", response.output)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
