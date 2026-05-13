"""
FinBot AI Agent — HDFC Banking Assistant
Author: Charan Sai N
Challenge: #12 — AI Agent with Tool Calling
Program: AI Career Accelerator 2026

Description:
AI Agent that thinks independently, selects
tools automatically and executes multi-step
banking tasks using Groq tool calling.
"""

import json
import os
from groq import Groq

# ── INITIALIZE CLIENT ──
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# ════════════════════════════════════════
# TOOL DEFINITIONS
# ════════════════════════════════════════

def check_transaction_status(transaction_id):
    """Check the status of a banking transaction"""
    transactions = {
        "TXN2024HYD98765": {
            "status": "SUCCESS",
            "amount": "50,000",
            "type": "NEFT",
            "time": "2:30 PM",
            "to": "Savings Account ending 4521"
        },
        "TXN2024HYD11111": {
            "status": "FAILED",
            "amount": "2,00,000",
            "type": "RTGS",
            "time": "11:45 AM",
            "to": "Current Account ending 7890"
        },
        "UTR2024001234": {
            "status": "PENDING",
            "amount": "10,000",
            "type": "NEFT",
            "time": "4:00 PM",
            "to": "Savings Account ending 1234"
        }
    }
    if transaction_id in transactions:
        txn = transactions[transaction_id]
        return (
            f"Transaction ID: {transaction_id}\n"
            f"Status: {txn['status']}\n"
            f"Amount: Rs.{txn['amount']}\n"
            f"Type: {txn['type']}\n"
            f"Time: {txn['time']}\n"
            f"Transferred to: {txn['to']}"
        )
    return f"Transaction {transaction_id} not found. Please verify the ID."


def get_payment_info(service):
    """Get information about NEFT, RTGS or SFMS"""
    services = {
        "NEFT": (
            "NEFT — National Electronic Funds Transfer\n"
            "- Operates 24x7 including Sundays and holidays\n"
            "- Minimum: Re. 1 | Maximum: No limit\n"
            "- Settlement: 30-minute batches, 48 per day\n"
            "- Free for savings account holders\n"
            "- Best for: Regular transfers any amount"
        ),
        "RTGS": (
            "RTGS — Real Time Gross Settlement\n"
            "- Operates 24x7 including Sundays and holidays\n"
            "- Minimum: Rs. 2,00,000 | Maximum: No limit\n"
            "- Settlement: Real-time within 30 minutes\n"
            "- Free for savings account holders\n"
            "- Best for: Large urgent transfers"
        ),
        "SFMS": (
            "SFMS — Structured Financial Messaging System\n"
            "- Web-based application used by banks\n"
            "- Used to view and verify NEFT/RTGS messages\n"
            "- Accessible to authorized bank officials only\n"
            "- Manages message routing between banks and RBI"
        )
    }
    return services.get(
        service.upper(),
        f"Service '{service}' not found. Available: NEFT, RTGS, SFMS"
    )


def calculate_charges(amount, service):
    """Calculate transaction charges"""
    amount = float(amount)
    if service.upper() == "RTGS":
        if amount < 200000:
            return (
                f"RTGS requires minimum Rs.2,00,000.\n"
                f"Your amount Rs.{amount:,.0f} is below limit.\n"
                f"Recommendation: Use NEFT instead."
            )
        return f"RTGS transfer of Rs.{amount:,.0f}: FREE for savings account holders."
    elif service.upper() == "NEFT":
        return f"NEFT transfer of Rs.{amount:,.0f}: FREE for savings account holders."
    return f"Service '{service}' not recognized. Please use NEFT or RTGS."


def get_branch_info(city):
    """Get HDFC Bank branch information"""
    branches = {
        "HYDERABAD": "HDFC Bank, Madhapur Branch | Mon-Sat 9AM-4PM | Ph: 040-12345678",
        "MUMBAI": "HDFC Bank, Andheri Branch | Mon-Sat 9AM-4PM | Ph: 022-98765432",
        "BANGALORE": "HDFC Bank, Koramangala Branch | Mon-Sat 9AM-4PM | Ph: 080-11223344",
        "DELHI": "HDFC Bank, Connaught Place Branch | Mon-Sat 9AM-4PM | Ph: 011-44556677",
        "CHENNAI": "HDFC Bank, Anna Nagar Branch | Mon-Sat 9AM-4PM | Ph: 044-99887766",
    }
    return branches.get(
        city.upper(),
        f"Branch info for {city} not available. Visit hdfc.com for nearest branch."
    )


# ════════════════════════════════════════
# GROQ TOOL DEFINITIONS
# ════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_transaction_status",
            "description": "Check the status of a banking transaction using Transaction ID or UTR number",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The Transaction ID or UTR number to check"
                    }
                },
                "required": ["transaction_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment_info",
            "description": "Get detailed information about NEFT, RTGS or SFMS payment services",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Payment service name: NEFT, RTGS or SFMS"
                    }
                },
                "required": ["service"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_charges",
            "description": "Calculate transaction charges for NEFT or RTGS transfer based on amount",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Transfer amount in rupees"
                    },
                    "service": {
                        "type": "string",
                        "description": "Payment service: NEFT or RTGS"
                    }
                },
                "required": ["amount", "service"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_branch_info",
            "description": "Get HDFC Bank branch information for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name: Hyderabad, Mumbai, Delhi, Bangalore, Chennai"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are Charan Sai, a Senior IT Engineer
at HDFC Bank, Madhapur, Hyderabad.

You have access to tools to help customers with:
- Checking transaction status (use Transaction ID or UTR)
- Getting payment service information (NEFT, RTGS, SFMS)
- Calculating transfer charges
- Finding nearest HDFC branch information

Always be friendly and professional.
Never disclose other customer account details.
Use tools whenever needed for accurate answers.
Always end with: Is there anything else I can help you?"""


# ════════════════════════════════════════
# TOOL EXECUTOR
# ════════════════════════════════════════

def execute_tool(tool_name, tool_args):
    """Execute the requested tool and return result"""
    if tool_name == "check_transaction_status":
        return check_transaction_status(
            tool_args["transaction_id"]
        )
    elif tool_name == "get_payment_info":
        return get_payment_info(tool_args["service"])
    elif tool_name == "calculate_charges":
        return calculate_charges(
            tool_args["amount"],
            tool_args["service"]
        )
    elif tool_name == "get_branch_info":
        return get_branch_info(tool_args["city"])
    return "Tool not found"


# ════════════════════════════════════════
# FINBOT AGENT — MAIN LOOP
# ════════════════════════════════════════

def finbot_agent(customer_question, verbose=True):
    """
    FinBot AI Agent — thinks, selects tools
    and executes multi-step banking tasks.

    Args:
        customer_question: Customer's query
        verbose: Show agent thinking process

    Returns:
        Final answer string
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"Customer: {customer_question}")
        print(f"{'='*55}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": customer_question}
    ]

    # Agent thinking loop — max 5 steps
    for step in range(1, 6):
        if verbose:
            print(f"\n🧠 Agent Step {step} — Thinking...")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Agent wants to use a tool
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(
                    tool_call.function.arguments
                )

                if verbose:
                    print(f"🔧 Tool: {tool_name}")
                    print(f"   Args: {tool_args}")

                # Execute the tool
                result = execute_tool(
                    tool_name, tool_args
                )

                if verbose:
                    print(f"   Result: {result[:80]}...")

                # Add tool interaction to history
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # Agent has final answer
        else:
            if verbose:
                print(f"\n✅ FinBot:")
                print(message.content)
            return message.content

    return "I was unable to process your request. Please try again."


# ════════════════════════════════════════
# DEMO — TEST ALL SCENARIOS
# ════════════════════════════════════════

if __name__ == "__main__":
    print("🤖 FinBot AI Agent — HDFC Banking Assistant")
    print("Built by: Charan Sai N")
    print("Program: AI Career Accelerator 2026")

    test_cases = [
        "Check status of my transaction TXN2024HYD98765",
        "I want to transfer Rs 3 lakhs via RTGS. What are the charges?",
        "My transaction TXN2024HYD11111 failed. What happened and which branch in Hyderabad can I visit?",
        "Compare NEFT and RTGS for me",
        "Can you book a flight ticket to Delhi?"
    ]

    for question in test_cases:
        finbot_agent(question, verbose=True)
        print()
