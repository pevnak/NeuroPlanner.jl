(define (problem s2)
	(:domain sokoban)
	(:objects sokoban1 sokoban2 crate1 crate2 l1 l2 l5 l6 l9 l10 l11 l12 l13 l14 l15 l16 l17 l18)
	(:init (sokoban sokoban1) 
		   (sokoban sokoban2)
		   (crate crate1)	
		   (crate crate2)
		   
		   ;;horizontal relationships
		   (leftOf l1 l2) 
		   (leftOf l5 l6) 
		   (leftOf l9 l10) (leftOf l10 l11) (leftOf l11 l12) 
 		   (leftOf l13 l14) (leftOf l14 l15) (leftOf l15 l16)
 		   (leftOf l17 l18)

 		   ;;vertical relationships
 		   (below l5 l1) (below l6 l2)
 		   (below l9 l5) (below l10 l6)
 		   (below l13 l9) (below l14 l10) (below l15 l11) (below l16 l12)
 		   (below l17 l13) (below l18 l14)

 		   ;;initialize sokoban and crate
		   (at sokoban1 l10)
		   (at sokoban2 l16)
		   (at crate1 l9)
 		   (at crate2 l15) 

 		   ;;clear spaces
		   (clear l1) 
		   (clear l2) 
		   (clear l5) 
		   (clear l6) 
		   (clear l11)
		   (clear l12) 
		   (clear l13) 
		   (clear l14)
		   (clear l17)   				
		   (clear l18))

	(:goal (and (at crate1 l9) (at crate2 l2)))
)