        module TensProduct
        implicit none
        
        contains        
!       Print real part of a complex matrix on a file        
        subroutine MatPrintCR(A, nr, nc, filename, extra, message) 
!           Scalar variables        
            integer :: nr, nc
            character :: extra !To print extra information
!           Array variables        
            complex, dimension(:,:), allocatable :: A 
            character(:), allocatable :: filename
            character(:), allocatable, intent(in), optional :: message
!           Local scalars  
            integer :: ii, jj
            real :: norm
!           Local array 
            character(:), allocatable :: loc_mes, def_mes
            
            def_mes='   '
            if (present(message)) then
                loc_mes = message
            else
                loc_mes = def_mes
            end if
            
            open(10, FILE=filename, STATUS='unknown', ACCESS='append')
            if(extra=='Y') write(10,*)  loc_mes
            if(extra=='Y') write(10,*) 'Dimensions: (', nr, ',', nc, ')'
            do  ii=1,nr
                write(10, *) (real(A(ii,jj)), jj=1,nc)
            end do
            if(extra=='Y') write(10,*) '    '
            close(10)
        end subroutine

!       Print a real vector as row on a file       
        subroutine VecPrintR(a, nn, filename, extra, message) 
!           Scalar variables        
            integer :: nn
            character :: extra !To print extra information
!           Array variables        
            real, dimension(:), allocatable :: a 
            character(:), allocatable :: filename
            character(:), allocatable, intent(in), optional :: message
!           Local scalars  
            integer :: ii
!           Local array 
            character(:), allocatable :: loc_mes, def_mes
            
            def_mes='   '
            if (present(message)) then
                loc_mes = message
            else
                loc_mes = def_mes
            end if
            
            open(10, FILE=filename, STATUS='unknown', ACCESS='append')
            if(extra=='Y') write(10,*)  loc_mes
            if(extra=='Y')write(10,*) 'Dimension: ', nn
            write(10, *) (a(ii), ii=1,nn)
            if(extra=='Y') write(10,*) '    '
            close(10)
        end subroutine  
        
!       Performs the tensor product between two matices A(x)B        
        subroutine TensProd(A, ra, ca, B, rb, cb, C)
!           Scalar variables        
            integer :: ra, ca, rb, cb
!           Array variables        
            complex, dimension(:,:), allocatable :: A, B, C 
!           Local scalars  
            integer :: ii, jj, kk, ll 
            
            allocate(C(ra*rb,ca*cb))
            do ii=1,rb
                do jj=1,cb
                    do kk=1,ra
                        do ll=1,ca
                        C(ra*(ii-1)+kk,ca*(jj-1)+ll)=A(kk,ll)*B(ii,jj)
                        end do
                    end do
                end do
            end do
        end subroutine
            
!       Performs the tensor product with identity on right       
        subroutine TensProdIDR(A, ra, ca, NID, C)
!           Scalar variables        
            integer :: ra, ca, NID
!           Array variables        
            complex, dimension(:,:), allocatable :: A, C 
!           Local scalars  
            integer :: ii, kk, ll 
            
            allocate(C(ra*NID,ca*NID))
            C=0.0
            do ii=1,NID
                do kk=1,ra
                    do ll=1,ca
                    C(ra*(ii-1)+kk,ca*(ii-1)+ll)=A(kk,ll)
                    end do
                end do
            end do
        end subroutine
        
!       Performs the tensor product with identity on left       
        subroutine TensProdIDL(NID, B, rb, cb, C)
!           Scalar variables        
            integer :: NID, rb, cb
!           Array variables        
            complex, dimension(:,:), allocatable :: B, C 
!           Local scalars  
            integer :: ii, jj, kk, ll 
            
            allocate(C(NID*rb,NID*cb))
            C=0.0
            do ii=1,rb
                do jj=1,cb
                    do kk=1,NID
                        C(NID*(ii-1)+kk,NID*(jj-1)+kk)=B(ii,jj)
                    end do
                end do
            end do
        end subroutine
        
!       Swap MNEW and MOLD
        subroutine SwapON(MOLD, MNEW, ii)
!           Array variables  
            complex, dimension(:,:), allocatable :: MOLD, MNEW
!           Scalar variables        
            integer :: ii
            
            deallocate(MOLD)
            allocate(MOLD(2**ii,2**ii))
            MOLD=MNEW
            deallocate(MNEW)
        end subroutine
        
        end module
   
   
        
        program IsingModel
        use TensProduct
        implicit none 
!       Scalar variables        
        integer :: NN, IS, N2, KK
        real :: lambda
!       Array variables  
        complex, dimension(:,:), allocatable :: ID2, SX, SZ, HN
        real, dimension(:), allocatable :: ev 
!       Local scalars  
        integer :: ii, info1, lwork
        logical :: DB, DB2
!       Local arrays  
        complex, dimension(:,:), allocatable :: MOLD, MNEW
        character(:), allocatable :: checkfile, mex1, mex2
        complex, dimension(:), allocatable :: work 
        real, dimension(:), allocatable :: rwork 

        
!       Initilaize debug valiables
        DB=.false.
        checkfile='checks.txt'
        mex1='Hamiltonian matrix'
        mex2='Hamiltonian matrix (diagonalized)'

!       Initialize Pauli matrices and 2x2 Identity        
        allocate(ID2(2,2))
        allocate(SX(2,2))
        allocate(SZ(2,2))
        ID2=0.0
        ID2(1,1)=1.0
        ID2(2,2)=1.0
        SX=0.0
        SX(1,2)=1.0
        SX(2,1)=1.0
        SZ=0.0
        SZ(1,1)=1.0
        SZ(2,2)=-1.0
        
!       Open input file and read parameters        
        open(77, FILE='parameters.txt', STATUS='old')
        read(77,*) NN, lambda, KK
        close (77, STATUS='keep')

!       Initialize hamiltonian matrix
        N2=2**NN
        allocate(HN(N2, N2))
        HN=0.0

!       First step external field term
        allocate(MOLD(2,2))
        MOLD=SZ
        do ii=2,NN
            call TensProdIDR(MOLD, 2**(ii-1), 2**(ii-1), 2, MNEW) 
            call SwapON(MOLD, MNEW, ii)
        end do
        HN=HN+MOLD
        deallocate(MOLD)        

!       Recursive steps external field term                
        do IS=2, NN
            allocate(MOLD(2,2))
            MOLD=ID2
            do ii=2,IS-1
                call TensProdIDL(2, MOLD, 2**(ii-1), 2**(ii-1), MNEW) 
                call SwapON(MOLD, MNEW, ii)
            end do
            call TensProd(MOLD, 2**(IS-1), 2**(IS-1), SZ, 2, 2, MNEW)
            call SwapON(MOLD, MNEW, IS)
            do ii=IS+1,NN
                call TensProdIDR(MOLD, 2**(ii-1), 2**(ii-1), 2, MNEW) 
                call SwapON(MOLD, MNEW, ii)
            end do
            HN=HN+MOLD
            deallocate(MOLD)
        end do

!       First step coupling term        
        call TensProd(SX, 2, 2, SX, 2, 2, MOLD)
        do ii=3,NN
            call TensProdIDR(MOLD, 2**(ii-1), 2**(ii-1), 2, MNEW) 
            call SwapON(MOLD, MNEW, ii)
        end do
        HN=HN+lambda*MOLD
        deallocate(MOLD)        

!       Recursive steps coupling term                 
        do IS=2, NN-1
            allocate(MOLD(2,2))
            MOLD=ID2
            do ii=2,IS-1
                call TensProdIDL(2, MOLD, 2**(ii-1), 2**(ii-1), MNEW) 
                call SwapON(MOLD, MNEW, ii)
            end do
            call TensProd(MOLD, 2**(IS-1), 2**(IS-1), SX, 2, 2, MNEW)
            call SwapON(MOLD, MNEW, IS)
            call TensProd(MOLD, 2**IS, 2**IS, SX, 2, 2, MNEW)
            call SwapON(MOLD, MNEW, IS)
            do ii=IS+2,NN
                call TensProdIDR(MOLD, 2**(ii-1), 2**(ii-1), 2, MNEW) 
                call SwapON(MOLD, MNEW, ii)
            end do
            HN=HN+lambda*MOLD
            deallocate(MOLD)
        end do
        if(DB) call MatPrintCR(HN, N2, N2, checkfile, 'Y', mex1)
        
!       Diagonalize the hamiltonian matrix and store the eigenvalues
        allocate(ev(N2))
        lwork=2*(N2)
        allocate(work(max(1,lwork)))
        allocate(rwork(max(1,3*N2-2)))
        call cheev('N', 'U', N2, HN, N2, ev, work, lwork, rwork, info1)
        if(DB) call MatPrintCR(HN, N2, N2, checkfile, 'Y', mex2)

!       Save the ground state eigenvectors and eigenvalues on file        
!        open(10, FORM='unformatted', 
!     &            FILE='evec.bin', STATUS='unknown', ACCESS='append')
!        write(10) complex(lambda,0.0), (HN(ii,1), ii=1,N2)
!        close(10)
        
        if(DB) then
        open(10, FILE='evec.txt', STATUS='unknown', ACCESS='append')
        write(10, *) complex(lambda,0.0), (HN(ii,1), ii=1,N2)
        close(10)
        end if
        
        
        open(10, FORM='unformatted', 
     &            FILE='eval.bin', STATUS='unknown', ACCESS='append')
        write(10) lambda, ev(1)
        close(10)

!       Free memory
        deallocate(HN)
        deallocate(ev)
        
        end program


        



