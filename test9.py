import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import json

# To jest nasza funkcja-narzędzie. W prawdziwej aplikacji łączyłaby się z API pogodowym.
def get_current_weather(location):
    """Zwraca informacje o pogodzie dla podanej lokalizacji."""
    if "warszawa" in location.lower():
        return json.dumps({"location": "Warszawa", "temperature": "18", "forecast": "słonecznie"})
    elif "londyn" in location.lower():
        return json.dumps({"location": "Londyn", "temperature": "15", "forecast": "deszcz"})
    else:
        return json.dumps({"location": location, "temperature": "nieznana", "forecast": "brak danych"})

# Opis narzędzia dla modelu
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Pobierz aktualne informacje o pogodzie w danej lokalizacji",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Miasto, dla którego ma być sprawdzona pogoda, np. Warszawa",
                    },
                },
                "required": ["location"],
            },
        }
    }
]

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_api_key"))



def run_conversation(user_prompt: str):
    messages = [{"role": "user", "content": user_prompt}]

    print(f"Użytkownik: {user_prompt}")

    # Pierwsze wywołanie modelu
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message

    # Sprawdzenie, czy model chce użyć narzędzia
    tool_calls = response_message.tool_calls
    if tool_calls:
        # Dodanie odpowiedzi asystenta (z prośbą o wywołanie narzędzia) do historii
        messages.append(response_message)

        # Dostępne funkcje
        available_functions = {
            "get_current_weather": get_current_weather,
        }

        # Wywołanie wszystkich narzędzi, o które prosi model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            print(f"Model prosi o wywołanie: {function_name}({function_args})")

            function_response = function_to_call(**function_args)

            # Dodanie wyniku działania narzędzia do historii
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        # Drugie wywołanie modelu, tym razem z wynikiem z narzędzia
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        final_response = second_response.choices[0].message.content
        print(f"Agent: {final_response}")
    else:
        # Jeśli model nie użył narzędzia, po prostu zwracamy jego odpowiedź
        final_response = response_message.content
        print(f"Agent: {final_response}")


# Test agenta
run_conversation("Jaka jest teraz pogoda w Londynie?")
print("\n---\n")
run_conversation("Cześć, jak się masz?")