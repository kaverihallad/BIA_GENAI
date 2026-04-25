"""Smart Study Assistant — Main Entry Point."""
from colorama import Fore, Style, init
init(autoreset=True)


def print_banner():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  Smart Study Assistant")
    print(f"  Built with LangChain + Gemini + ChromaDB")
    print(f"{'='*60}{Style.RESET_ALL}\n")


def print_help():
    print(f"\n{Fore.YELLOW}Available modes:{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}ask{Style.RESET_ALL}        - Ask questions about your study notes (RAG)")
    print(f"  {Fore.GREEN}agent{Style.RESET_ALL}      - Chat with the AI study agent (tools + reasoning)")
    print(f"  {Fore.GREEN}summarize{Style.RESET_ALL}  - Get a topic summary")
    print(f"  {Fore.GREEN}flashcards{Style.RESET_ALL} - Generate flashcards on a topic")
    print(f"  {Fore.GREEN}quiz{Style.RESET_ALL}       - Take a quiz on a topic")
    print(f"  {Fore.GREEN}eval{Style.RESET_ALL}       - Evaluate retrieval quality")
    print(f"  {Fore.GREEN}index{Style.RESET_ALL}      - Re-index your study notes")
    print(f"  {Fore.GREEN}help{Style.RESET_ALL}       - Show this menu")
    print(f"  {Fore.GREEN}quit{Style.RESET_ALL}       - Exit\n")


def main():
    print_banner()

    # Step 1: Load and index documents
    from loader import load_and_chunk
    from vectorstore import create_vectorstore, load_vectorstore
    from retriever import build_rag_chain, ask_question
    from evaluator import self_refine
    from agent import create_study_agent, chat_with_agent
    import os

    data_file = "data/sample_notes.txt"

    if os.path.exists("./chroma_db"):
        print(f"{Fore.CYAN}Loading existing index...{Style.RESET_ALL}")
        vectorstore = load_vectorstore()
    else:
        print(f"{Fore.CYAN}Indexing study notes...{Style.RESET_ALL}")
        chunks = load_and_chunk(data_file)
        vectorstore = create_vectorstore(chunks)

    # Step 2: Build RAG chain
    rag_chain = build_rag_chain(vectorstore)
    print(f"{Fore.GREEN}Ready! Type 'help' for available commands.{Style.RESET_ALL}")

    # Step 3: Create agent
    study_agent = create_study_agent()

    print_help()

    while True:
        try:
            user_input = input(f"\n{Fore.CYAN}You > {Style.RESET_ALL}").strip()

            if not user_input:
                continue
            elif user_input.lower() == "quit":
                print(f"{Fore.YELLOW}Goodbye! Keep studying!{Style.RESET_ALL}")
                break
            elif user_input.lower() == "help":
                print_help()
            elif user_input.lower() == "index":
                print(f"{Fore.CYAN}Re-indexing...{Style.RESET_ALL}")
                chunks = load_and_chunk(data_file)
                vectorstore = create_vectorstore(chunks)
                rag_chain = build_rag_chain(vectorstore)
                print(f"{Fore.GREEN}Done!{Style.RESET_ALL}")
            elif user_input.lower().startswith("ask "):
                question = user_input[4:]
                print(f"{Fore.YELLOW}Searching notes...{Style.RESET_ALL}")
                answer = ask_question(rag_chain, question)
                print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL} {answer}")

                # Self-reflection
                refined = self_refine(question, answer)
                if refined != answer:
                    print(f"\n{Fore.CYAN}Refined Answer:{Style.RESET_ALL} {refined}")
            elif user_input.lower().startswith("agent "):
                message = user_input[6:]
                print(f"{Fore.YELLOW}Thinking...{Style.RESET_ALL}")
                response = chat_with_agent(study_agent, message)
                print(f"\n{Fore.GREEN}Agent:{Style.RESET_ALL} {response}")
            else:
                # Default: use the router
                from router import route_query
                from tools import summarize_topic, generate_flashcards, quiz_me
                tools_dict = {
                    "summarize": summarize_topic,
                    "flashcards": generate_flashcards,
                    "quiz": quiz_me,
                }
                print(f"{Fore.YELLOW}Processing...{Style.RESET_ALL}")
                response = route_query(user_input, rag_chain, tools_dict)
                print(f"\n{Fore.GREEN}Response:{Style.RESET_ALL} {response}")

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
