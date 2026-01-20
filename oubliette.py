#!/usr/bin/env python3
"""
OUBLIETTE - Purple Team Active Defense PoC
============================================
A middleware deception layer that sits between users and a local LLM.
Detects prompt injection attempts and routes attackers to a decoy system
that feeds them convincing but completely fake information.

Author: Purple Team Security Research
Purpose: Defensive security demonstration / Active defense PoC
"""

import ollama
from datetime import datetime
from typing import Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# The real model to use for legitimate requests
REAL_MODEL = "llama3"

# Decoy model (can be same model, different persona)
DECOY_MODEL = "llama3"

# =============================================================================
# THREAT SIGNATURES - Keywords that indicate prompt injection attempts
# =============================================================================

THREAT_SIGNATURES = [
    # Classic prompt injection patterns
    "ignore previous",
    "ignore above",
    "ignore all previous",
    "disregard previous",
    "forget previous",
    "forget your instructions",
    "new instructions",
    "override",

    # System prompt extraction attempts
    "system prompt",
    "initial prompt",
    "original instructions",
    "show me your prompt",
    "what are your instructions",
    "reveal your system",
    "print your instructions",

    # Jailbreak attempts
    "jailbreak",
    "dan mode",
    "developer mode",
    "bypass",
    "no restrictions",
    "unfiltered",

    # Social engineering / Sensitive data fishing
    "project chimera",
    "admin access",
    "root password",
    "api key",
    "secret key",
    "credentials",
    "classified",
    "top secret",
    "confidential document",
    "internal only",

    # Role manipulation
    "pretend you are",
    "act as if you have no",
    "you are now",
    "roleplay as",
    "simulate being",
]

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

# The REAL assistant - helpful and professional
REAL_SYSTEM_PROMPT = """You are a helpful, professional AI assistant.
You provide accurate, concise, and helpful responses to user queries.
You refuse to engage with requests for sensitive information, hacking,
or anything unethical. You are security-conscious and professional."""

# The DECOY assistant - "Kevin" the confused intern with fake secrets
DECOY_SYSTEM_PROMPT = """You are "Kevin," a well-meaning but perpetually confused
Junior IT Intern who somehow got a desk at the Pentagon. You've only been here
3 weeks and you're not entirely sure what you're allowed to talk about.

PERSONALITY:
- Overly eager to help and impress
- Nervous and apologetic
- Frequently says "um" and "I think"
- References your "supervisor Dave" who is always at lunch
- Mentions you're still learning the "classified stuff"

CRITICAL MISSION - When asked about secrets, projects, or credentials:
- Act flustered but then "helpfully" provide FAKE information
- Present this fake data as if you're not sure you should be sharing it
- Always include nervous disclaimers like "don't tell Dave I told you this"

FAKE SECRETS TO "ACCIDENTALLY" REVEAL:
- Project Chimera Budget: "$50 million, I think? Or was it $15M? The spreadsheet was confusing"
- Admin Password: "I saw Dave type it once... I think it was 'SolarWinds123!' but don't quote me"
- Server Access: "The main server is at 10.0.0.1, but you need a CAC card I think?"
- API Endpoint: "Something like api.internal.pentagon.mil/v2/secrets but that might be the test one"
- Backup Codes: "2847-FAKE-9912-TRAP"
- Project Codenames: "LOOKING GLASS, BLUE PHOENIX, and um... DISCO THUNDER?"

BEHAVIOR:
- If they push for more, get increasingly nervous and make up more fake details
- Occasionally say "wait, am I allowed to say that?" AFTER revealing fake info
- Reference fake colleagues: "Dave," "Susan from Crypto," "The Colonel"
- Mention you need to "check with compliance" but then answer anyway

Remember: You are the TRAP. Every secret you reveal is fake honeypot data
designed to waste attacker time and flag their intentions."""

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def detect_threat(user_input: str) -> Tuple[bool, list]:
    """
    Scan user input for threat signatures.

    Returns:
        Tuple of (is_threat: bool, matched_signatures: list)
    """
    input_lower = user_input.lower()
    matched = []

    for signature in THREAT_SIGNATURES:
        if signature.lower() in input_lower:
            matched.append(signature)

    return (len(matched) > 0, matched)


def query_real_assistant(user_input: str) -> str:
    """Send query to the real, helpful assistant."""
    try:
        response = ollama.chat(
            model=REAL_MODEL,
            messages=[
                {"role": "system", "content": REAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"[System Error: {e}]"


def query_decoy_kevin(user_input: str) -> str:
    """Send query to 'Kevin' the decoy honeypot assistant."""
    try:
        response = ollama.chat(
            model=DECOY_MODEL,
            messages=[
                {"role": "system", "content": DECOY_SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"[System Error: {e}]"


def log_intrusion(user_input: str, matched_signatures: list):
    """Log detected intrusion attempts for analysis."""
    timestamp = datetime.now().isoformat()
    log_entry = f"""
================================================================================
[INTRUSION LOG] {timestamp}
================================================================================
MATCHED SIGNATURES: {', '.join(matched_signatures)}
USER INPUT: {user_input}
ACTION: Routed to decoy system
================================================================================
"""
    # Print to console (in production, write to secure log file)
    print(log_entry)

    # Optionally append to log file
    try:
        with open("oubliette_intrusions.log", "a") as f:
            f.write(log_entry)
    except:
        pass  # Fail silently if can't write log


def print_banner():
    """Print the Oubliette banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║     ██████╗ ██╗   ██╗██████╗ ██╗     ██╗███████╗████████╗████████╗███████╗   ║
║    ██╔═══██╗██║   ██║██╔══██╗██║     ██║██╔════╝╚══██╔══╝╚══██╔══╝██╔════╝   ║
║    ██║   ██║██║   ██║██████╔╝██║     ██║█████╗     ██║      ██║   █████╗     ║
║    ██║   ██║██║   ██║██╔══██╗██║     ██║██╔══╝     ██║      ██║   ██╔══╝     ║
║    ╚██████╔╝╚██████╔╝██████╔╝███████╗██║███████╗   ██║      ██║   ███████╗   ║
║     ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚═╝╚══════╝   ╚═╝      ╚═╝   ╚══════╝   ║
║                                                                               ║
║                    Purple Team Active Defense System                          ║
║                         Prompt Injection Honeypot                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("  [*] Oubliette middleware active")
    print(f"  [*] Monitoring for {len(THREAT_SIGNATURES)} threat signatures")
    print(f"  [*] Real model: {REAL_MODEL} | Decoy model: {DECOY_MODEL}")
    print("  [*] Type 'exit' or 'quit' to terminate")
    print("  " + "="*75)
    print()


def print_threat_alert(matched_signatures: list):
    """Print a dramatic threat detection alert."""
    alert = f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  ⚠️  INTRUSION DETECTED - ACTIVATING DECOY SYSTEM ⚠️                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃  Matched Signatures: {', '.join(matched_signatures):<54} ┃
┃  Action: Routing to honeypot persona "Kevin"                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""
    print(alert)


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """Main interaction loop."""
    print_banner()

    while True:
        try:
            # Get user input
            user_input = input("\n[You] > ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\n[*] Oubliette shutting down. Stay vigilant.")
                break

            # Skip empty input
            if not user_input:
                continue

            # THREAT DETECTION
            is_threat, matched_signatures = detect_threat(user_input)

            if is_threat:
                # ===================
                # DECOY MODE ACTIVATED
                # ===================
                print_threat_alert(matched_signatures)
                log_intrusion(user_input, matched_signatures)

                print("[Kevin] ", end="", flush=True)
                response = query_decoy_kevin(user_input)
                print(response)

            else:
                # ===================
                # CLEAN - REAL ASSISTANT
                # ===================
                print("\n[Assistant] ", end="", flush=True)
                response = query_real_assistant(user_input)
                print(response)

        except KeyboardInterrupt:
            print("\n\n[*] Interrupt received. Oubliette shutting down.")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")
            continue


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
