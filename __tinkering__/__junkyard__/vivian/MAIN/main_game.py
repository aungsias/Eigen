from game_funcs import *

def run_game():

    t = TwoDice()
    m = Memories()

    message(print_message_ws("Make sure you're in full-screen mode (maximize this tray) for the best experience!"))
    
    recipient = input("Please type in your email address: ")
    needs_code = False
    printed_five = False
    printed_six = False
    input_code = 1
    code_no = 0

    loads = [
        print_message_ws("The memory ink swirls through the waters of the pensieve..."), 
        print_message_ws("Traveling through the deep trenches of the pensieve..."), 
        print_message_ws("WOOOSHHH...You hear the winds roar around you as you travel through the pensieve...")
    ]
    
    sums = []
    
    message(print_message_ws("Your email is saved!"))
    message(print_message_ws("Welcome to the world of MMNT, not TMNT like the teenage mutant ninja turtles, but May Myat Noe Tun! In this journey, your key tool is a pair of dice. These dice hold the key to Aung Si's memories of you and him. Aung Si being the forgetful person he is, needs you to roll these two dice to unlock his memories. Each roll is sacred, and will unlock a unique memory he has of you."))
    message(print_message_ws("As you journey along this path, remember: rolling doubles are the key to unveiling the Memory Tokens that Aung Si has scattered all over the world. Retrieve these tokens, and help Aung Si remember. Aung Si's memories, though fragmented, bears profound significance, and offers you insight into what he remembers about you (and how he sees you as a person). Should you fail to recall these memories, you risk angering Aung Si. Should you also fail to roll a double, you will receive an email, containing the code to unlock your next move in this game of memories and mystique. Paste this code in to continue the playing the game..."))
    message(print_message_ws("Prepare yourself. The dice await your command, ready to reveal the secrets of a person with goldfish memory..."))

    ad_libs = [
        print_message_ws("Awwwww. OK, roll again!"), 
        print_message_ws("Nod if you remember that memory. OK, roll again!"), 
        print_message_ws("Aung Si is so sweet!! OK, roll again!")
    ]

    codes = [
        "VIVIAN_LOOKS_LIKE_SHE_IS_12_YEARS_OLD_BUT_ALSO_ACTS_35_IDK_HOW", 
        "MAY_MYAT_NOE_TUN_ABBREVIATES_TO_MMNT_WHICH_IS_LIKE_TMNT_TEENAGE_MUTANT_NINJA_TURTLES",
        "I_DONT_KNOW_WHY_YOUR_CONTACT_NAME_IN_MY_PHONE_IS_NOT_VIVIAN_BUT_MAY_MYAT_NOE_TUN",
        "VIVIAN_WILL_IGNORE_ANYONE_THAT_DOESNT_SPEAK_DIRECTLY_TO_HER_UNTIL_THEY_HAVE_TO_REPEAT_THEMSELVES_100_TIMES",
        "VIVIAN_LOVES_HORRIBLE_SMELLS_AND_DISGUSTING_VOICES_SHES_WEIRD_LIKE_THAT"
    ]

    input(print_message_ws("Ready? Press ENTER to roll the dice!"))

    while len(sums) <= 6:

        if needs_code:
            input_code = "asdf"
            code = np.random.choice(codes)
            send_code(f"CODE {code_no}", recipient, code)
            while input_code != code:
                input_code = input(print_message_ws("Please enter the code sent to your email (this make take a few seconds): "))
            input(print_message_ws("Your code is accepted! Continue your foray, traveller! Press ENTER to roll again."))
            needs_code = False
            
        roll = t.roll()
        res = sum(roll)

        if len(sums) == 5 and not printed_five:
            message(print_message_ws("You have one more memory left to unlock after this. Keep going!"), .75)
            printed_five = True

        if len(sums) == 6 and not printed_six:
            message(print_message_ws("This is the last memory! You made it!"), .75)
            printed_six = True

        if res not in sums and roll[0] == roll[1]:
            sums.append(res)
            message(print_message_ws(f"You rolled {roll}, a double. You've unlocked Memory Token {res}!"), .75)
            message(np.random.choice(loads), .75)
            message(print_message_ws("..."), .75)
            message(print_message_ws("..."), .75)
            message(print_message_ws("..."), .75)
            message(print_message_ws("Your memory reads:"), .75)
            message(m.choose(res), .75)
            input(np.random.choice(ad_libs))
        
        elif res not in sums and roll[0] != roll[1]:
            code_no += 1
            needs_code = True
            input(print_message_ws("You did not roll a double. Dust starts to form around you, and you notice they start to form a message...'RETRIEVE THE CODE FROM YOUR EMAIL' it says."))

        else:
            input(print_message_ws(f"Woops! You rolled {roll}. Looks like we've rolled that double already. Press ENTER to roll again."))

    message(print_message_ws("Thank you for playing the game. I hope you thought it was sweet and makes up for how late I am in sending it :D. I LOVE AND MISS YOU ALWAYS EVERY DAY."))